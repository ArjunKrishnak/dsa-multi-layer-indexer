"""Phase 4b: high-occupancy (split-K / FlashDecoding) sparse-fused kernel, with a
balanced indexer.

Two problems with the single-block-per-head-group kernel
(``cuda_fused_sparse_indexer.py``):
  1. decode launches only ``H/16 = 8`` blocks (~6% of an H100's SMs);
  2. the indexer lives in only ``Hidx/16 = 4`` head-groups, and each of those groups
     re-reads the *entire* IndexK -> 4x redundant memory traffic, and at high context
     those 4 head-groups' blocks contend with the attention-only blocks instead of
     overlapping inside them (so fused ~ serial once the GPU saturates).

This version fixes both:
  * grid = S * H/16 * nsplit (nsplit=16 -> 128 decode blocks), so decode fills the GPU;
  * attention warps process this block's slice of the top-k keys (partials -> combine
    kernel);
  * indexer warps process this block's disjoint slice of the full prefix and compute
    **all Hidx heads** for those keys, writing the final per-key logit directly. Every
     block owns a unique key-slice (slice_id = hg*nsplit + split), so IndexK is read
     exactly once (4x less traffic) and the indexer is balanced across all 128 blocks.

The indexer GEMM runs in **fp8 (e4m3)** via ``mma.sync.m16n8k32`` -- matching the
production DeepSeek indexer and roughly halving its tensor-core cost at long
context (the 128K bottleneck), where the bf16 indexer contended with the bf16
sparse-MLA on the same tensor cores. Per-row fp8 scales fold into the head weight
(q) and a per-key dequant scale (k). Validated against an fp8 reference (the same
fp8 values, exact f32 math); sparse-MLA stays bf16, bit-exact vs the torch ref.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.cpp_extension import load_inline

from lagged_indexer_dsa.config import named_config
from lagged_indexer_dsa.logging_utils import get_logger

logger = get_logger("cuda_fused_sparse_splitk")

CUDA_SRC = r"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <math_constants.h>
#include <cuda_pipeline.h>
using namespace nvcuda;

#define HB 16
#define BNA 32       // attention keys per tile (small tile -> small smem -> 2 blocks/SM)
#define BNI 64       // indexer keys per tile
#define DQK 576
#define DV 512
#define DIDX 128
#define NWARP 4
#define DCOL (DV / NWARP)   // 128 output cols per attention warp
#define NT_PV (DCOL / 8)    // 16 n-tiles of 8 cols
// Padded shared-memory row strides: the natural strides (576 bf16 = 1152 B,
// 128 fp8 = 128 B, ...) are multiples of the 128 B bank period, so every
// fragment load was 8-way bank-conflicted.
#define KLD 584      // Qsh/Ksh row stride (bf16)
#define SLD 36       // Ssh row stride (f32)
#define PLD 40       // Psh row stride (bf16)
#define IKLD 144     // IdxQsh/IdxKsh row stride (fp8 bytes)
#define ISLD (Hidx + 2)  // IdxSsh row stride (f32, runtime Hidx)

__device__ __forceinline__ void bar_sync(int id, int cnt) {
  asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(cnt) : "memory");
}

// radix bucket = top-8 bits of the sign-flipped fp16 encoding; must match
// convert_to_uint16 in tilelang_dsa/topk_selector.py bit-for-bit
__device__ __forceinline__ int sel_bucket16(float x) {
  unsigned short b = __half_as_ushort(__float2half(x));
  b = (x < 0.f) ? (unsigned short)(~b) : (unsigned short)(b | 0x8000);
  return (int)(((unsigned)b) >> 8);
}

__device__ __forceinline__ void mma_bf16_16x8x16(
    float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// A fragment (row-major 16x16): lane row = lane&15, col half = +8 for lanes>=16.
__device__ __forceinline__ void ldmatrix_a(
    unsigned& a0, unsigned& a1, unsigned& a2, unsigned& a3,
    const __nv_bfloat16* base, int ld, int row0, int col0, int lane) {
  const __nv_bfloat16* p = base + (row0 + (lane & 15)) * ld + col0 + ((lane & 16) ? 8 : 0);
  unsigned addr = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3) : "r"(addr));
}

// B fragment for QK (K row-major [key][d], keys are n): non-transposed x2.
__device__ __forceinline__ void ldmatrix_b_k(
    unsigned& b0, unsigned& b1,
    const __nv_bfloat16* base, int ld, int key0, int col0, int lane) {
  const __nv_bfloat16* p = base + (key0 + (lane & 7)) * ld + col0 + ((lane & 8) ? 8 : 0);
  unsigned addr = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(b0), "=r"(b1) : "r"(addr));
}

// B fragments for PV, two adjacent 8-col n-tiles per x4.trans (V row-major
// [key][col], keys are k).
__device__ __forceinline__ void ldmatrix_b_v_x4(
    unsigned& n0b0, unsigned& n0b1, unsigned& n1b0, unsigned& n1b1,
    const __nv_bfloat16* base, int ld, int k0, int col0, int lane) {
  const __nv_bfloat16* p = base + (k0 + ((lane & 8) ? 8 : 0) + (lane & 7)) * ld
                         + col0 + ((lane & 16) ? 8 : 0);
  unsigned addr = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(n0b0), "=r"(n0b1), "=r"(n1b0), "=r"(n1b1) : "r"(addr));
}

// e4m3 x4 (one 32-bit word, dims d..d+3) -> two f16x2 words: .x = (d, d+1), .y = (d+2, d+3).
// The conversion is exact, so the products the tensor core forms are the same
// as with fp8 operands.
__device__ __forceinline__ uint2 hp_e4m3x4_to_f16x4(unsigned w) {
  __half2_raw lo = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(w & 0xffffu), __NV_E4M3);
  __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(w >> 16), __NV_E4M3);
  uint2 r;
  r.x = (unsigned)lo.x | ((unsigned)lo.y << 16);
  r.y = (unsigned)hi.x | ((unsigned)hi.y << 16);
  return r;
}
__device__ __forceinline__ void mma_f16_16x8x16(
    float& c0, float& c1, float& c2, float& c3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// One k32 step of the fp8 indexer GEMM on native f16 tensor instructions:
// the 4 fp8 of a 32-bit operand word occupy k-slots (2tid, 2tid+1) [lo] and
// (8+2tid, 8+2tid+1) [hi] of one m16n8k16 -- the same permutation for A and B,
// so the products are exactly the fp8 products; accumulation stays inside the
// tensor core across every k step (fp32). Both kernels use this helper, so the
// paged and dense entries are bit-identical (gate G1).
__device__ __forceinline__ void hp_index_mma_k32(
    float& c0, float& c1, float& c2, float& c3,
    uint2 a0, uint2 a1, uint2 a2, uint2 a3, uint2 b0, uint2 b1) {
  mma_f16_16x8x16(c0, c1, c2, c3, a0.x, a1.x, a0.y, a1.y, b0.x, b0.y);   // dims 0..15 of the step
  mma_f16_16x8x16(c0, c1, c2, c3, a2.x, a3.x, a2.y, a3.y, b1.x, b1.y);   // dims 16..31
}
// Head reduction of one 8-head n-tile straight from the mma fragment: thread
// (gid, tid) holds heads (n0+2tid, n0+2tid+1) of keys gid (c0,c1) and gid+8
// (c2,c3). Per-thread FMA chains, then hp_index_reduce_lanes folds the 4 lanes
// of a key with an xor tree -- deterministic, identical in both kernels.
__device__ __forceinline__ void hp_index_reduce_ntile(
    float& accA, float& accB, float c0, float c1, float c2, float c3, float w0, float w1) {
  accA += (c0 > 0.0f ? c0 : 0.0f) * w0;
  accA += (c1 > 0.0f ? c1 : 0.0f) * w1;
  accB += (c2 > 0.0f ? c2 : 0.0f) * w0;
  accB += (c3 > 0.0f ? c3 : 0.0f) * w1;
}
__device__ __forceinline__ float hp_index_reduce_lanes(float a) {
  a += __shfl_xor_sync(0xffffffffu, a, 1);
  a += __shfl_xor_sync(0xffffffffu, a, 2);
  return a;
}

#define SELN 4096
__device__ void cascade_select_row(
    const float* __restrict__ Planes, const int* __restrict__ Indices,
    int* __restrict__ NextIdx, int s, int S, int nblk, int topk, int win,
    int Skv, float* vals, int* hist, int* subhist, int* tie_slots, int* scal);

extern "C" __global__ void __launch_bounds__(256, 2) fused_sparse_splitk_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KV,
    const int* __restrict__ Indices, const __nv_fp8_e4m3* __restrict__ IndexQ,
    const __nv_fp8_e4m3* __restrict__ IndexK, const float* __restrict__ IndexW,
    const float* __restrict__ IndexKScale,   // [Skv]  per-key fp8 dequant scale
    const float* __restrict__ TapW,          // [H] per-head tap weights (nullptr = uniform)
    float* __restrict__ PartAcc,   // [Nb, 16, 512]
    float* __restrict__ PartM,     // [Nb, 16]
    float* __restrict__ PartL,     // [Nb, 16]
    float* __restrict__ SelSlab,   // [S, Skv]  (per-key logit, all heads summed)
    int* __restrict__ SelHist,     // [S, 256]  (with_select==2: radix histogram, zeroed)
    int* __restrict__ NextIdx,     // [S, topk] (with_select==4: folded select out)
    int* __restrict__ SelCounter,  // [S]       (with_select==4: election, zeroed)
    int S, int Skv, int H, int Hidx, int topk, int attend_k, int nsplit,
    float sm_scale, int with_select, int attn_on, int win, long kv_stride,
    long idx_k_stride, long idx_scale_stride,
    const int* __restrict__ SeqLens,     // [S] per-row prefix length (nullptr = Skv)
    const int* __restrict__ PageTable,   // [S, pt_stride] index-K page ids (nullptr = dense)
    long pt_stride, long page_bytes, long slab_stride,
    int nsplit_attn) {                   // attention-bearing splits per row (0 = nsplit)
  // attend_k <= topk enables pool/attend decoupling: Indices holds a
  // score-sorted candidate POOL of `topk` keys, but only the leading
  // attend_k columns participate in softmax/PV.  The pool's tail rides the
  // same QK tiles as score-only selection candidates — the generalization
  // of the admission window's "scored but not attended" semantics from
  // recency-picked keys to arbitrary tap-retained ones.  attend_k == topk
  // is the undecoupled kernel, bit-identical to its previous behavior.
  //
  // kv_stride == 0: legacy single-sequence semantics — the S query rows are
  // the trailing S positions of one KV sequence (q_pos = Skv-S+s_i).
  // kv_stride > 0: batched-decode semantics — row s_i is an independent
  // sequence whose KV lives at KV + s_i*kv_stride (elements), every row's
  // query sits at the same position q_pos = Skv-1, and Indices/planes are
  // per-row exactly as before.  Per-row arithmetic is unchanged, so a
  // batched launch is bitwise-equal to S legacy launches on the row slices.
  extern __shared__ unsigned char smem[];
  __nv_bfloat16* Qsh = (__nv_bfloat16*)smem;
  __nv_bfloat16* Ksh = Qsh + 16*KLD;
  float* Ssh   = (float*)(Ksh + BNA*KLD);
  __nv_bfloat16* Psh = (__nv_bfloat16*)(Ssh + 16*SLD);
  float* m_i   = (float*)(Psh + 16*PLD);
  float* l_i   = m_i + 16;
  float* corr_sh = l_i + 16;                                  // 16 (per-head rescale factor)
  float* tapw_sh = corr_sh + 16;                              // HB (cascade per-head tap weights)
  float* sel_w = tapw_sh + HB;                                // Hidx (folded q-scale * weight)
  // fp8 indexer scratch: IdxQ/IdxK are e4m3 (half the bf16 footprint).
  // IdxKsh is double-buffered (2 tiles) so the next tile's load overlaps the
  // current tile's mma+reduce -- the indexer is latency-bound on the per-tile load.
  __nv_fp8_e4m3* IdxQsh = (__nv_fp8_e4m3*)(sel_w + Hidx);     // Hidx*IKLD
  __nv_fp8_e4m3* IdxKsh = IdxQsh + Hidx*IKLD;                 // 2*BNI*IKLD (ping-pong)
  float* IdxSsh = (float*)(IdxKsh + 2*BNI*IKLD);              // BNI*ISLD (key-major)
  int* sel_idx = (int*)(IdxSsh + BNI*ISLD);                   // BNA
  int* s_hist = sel_idx + BNA;                                // 256 logit-histogram bins

  int nblk = H / HB;
  int t = threadIdx.x;
  int warp = t >> 5;
  int lane = t & 31;
  int split = blockIdx.x % nsplit;
  int tmp = blockIdx.x / nsplit;
  int hg = tmp % nblk;
  int s_i = tmp / nblk;
  // Per-row prefix length: SeqLens (device) overrides the host Skv so one
  // CUDA-graph captured launch replays correctly as the sequence grows and
  // across ragged batch rows; Skv stays the buffer CAPACITY (and the causal
  // sentinel value, which must exceed every q_pos).
  int skv_row = (SeqLens != nullptr) ? SeqLens[s_i] : Skv;
  int q_pos = (kv_stride > 0) ? (skv_row - 1) : (skv_row - S) + s_i;
  long slab_row = (slab_stride > 0) ? slab_stride : (long)Skv;
  const __nv_bfloat16* KVr = KV + (long)s_i * kv_stride;
  // with_select>=3 (tap cascade): the fp8 indexer never runs — selection is a
  // second consumer of the attention warps' own QK scores (incumbent top-k
  // cols + a score-only pass over the admission window), stored per
  // head-group into SelSlab reinterpreted as [nblk, S, topk+win].
  // with_select==4 additionally folds the top-k select into this launch: the
  // last CTA to finish a query row's planes runs the radix select inline.
  int do_sel = (with_select >= 3) ? 0 : with_select;
  int slice_id = hg * nsplit + split;       // 0 .. nblk*nsplit-1 (disjoint key slices)
  int n_slices = nblk * nsplit;

  for (int i = t; i < HB * DQK; i += blockDim.x) {
    int h = i / DQK, d = i % DQK;
    Qsh[h * KLD + d] = Q[(((long)s_i * H) + hg * HB + h) * DQK + d];
  }
  for (int i = t; i < HB; i += blockDim.x) { m_i[i] = -CUDART_INF_F; l_i[i] = 0.0f; }
  if (with_select >= 3) {
    // Distilled tap: tau = sum_h w_h ReLU(S_h). Weights are per GLOBAL head
    // (this block covers heads [hg*HB, hg*HB+HB)); nullptr keeps the legacy
    // uniform sum (a positive rescale, ranking-identical to the 1/H mean).
    for (int i = t; i < HB; i += blockDim.x)
      tapw_sh[i] = (TapW != nullptr) ? TapW[hg * HB + i] : 1.0f;
  }
  if (do_sel) {
    for (int i = t; i < Hidx; i += blockDim.x) sel_w[i] = IndexW[(long)s_i * Hidx + i];
    for (int i = t; i < Hidx * DIDX; i += blockDim.x) {
      int h = i / DIDX, d = i % DIDX;
      IdxQsh[h * IKLD + d] = IndexQ[(((long)s_i * Hidx) + h) * DIDX + d];
    }
    if (with_select == 2) {
      for (int i = t; i < 256; i += blockDim.x) s_hist[i] = 0;
    }
  }
  __syncthreads();

  if (warp < NWARP) {
    // attention over this split's slice of the top-k columns [c_lo, c_hi);
    // PV accumulator lives in registers (this warp's DCOL cols x 16 rows).
    int gid = lane >> 2;   // mma group row
    int tig = lane & 3;    // thread-in-group
    float po[NT_PV][4];
    #pragma unroll
    for (int nt = 0; nt < NT_PV; ++nt) po[nt][0] = po[nt][1] = po[nt][2] = po[nt][3] = 0.0f;
    float scale2 = sm_scale * 1.44269504f;   // log2-domain softmax
    // Decoupled split factors: the first nsplit_attn splits of a row carry
    // attention over the topk keys; every split carries an indexer slice. The
    // idle attention halves keep m=-inf / l=0 (neutral for the combine) and
    // skip the PartAcc write, so the indexer can be spread over many more
    // blocks without paying attention's per-block cost.
    int nsa = (nsplit_attn > 0) ? nsplit_attn : nsplit;
    int attn_here = attn_on && (split < nsa);
    int kps = (topk + nsa - 1) / nsa;
    int c_lo = split * kps;
    int c_hi = min(c_lo + kps, topk);
    int n_tiles = attn_here ? (max(c_hi - c_lo, 0) + BNA - 1) / BNA : 0;
    for (int kt = 0; kt < n_tiles; ++kt) {
      int col0 = c_lo + kt * BNA;
      for (int i = t; i < BNA; i += NWARP * 32) {
        int col = col0 + i;
        sel_idx[i] = (col < c_hi) ? Indices[(long)s_i * topk + col] : Skv;
      }
      bar_sync(1, NWARP * 32);
      for (int c = t; c < BNA * (DQK / 8); c += NWARP * 32) {
        int key = c / (DQK / 8);
        int chunk = c % (DQK / 8);
        int gidx = sel_idx[key];
        int valid = (gidx <= q_pos);
        long load_idx = valid ? (long)gidx : 0L;
        __pipeline_memcpy_async((int4*)Ksh + key * (KLD / 8) + chunk,
                                (const int4*)(KVr + load_idx * DQK) + chunk, sizeof(int4));
      }
      __pipeline_commit();
      __pipeline_wait_prior(0);
      bar_sync(1, NWARP * 32);
      // QK: this warp owns keys [warp*8, +8) = one n-tile. Raw scores to Ssh.
      {
        float qs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int kk0 = 0; kk0 < DQK; kk0 += 16) {
          unsigned a0, a1, a2, a3, b0, b1;
          ldmatrix_a(a0, a1, a2, a3, Qsh, KLD, 0, kk0, lane);
          ldmatrix_b_k(b0, b1, Ksh, KLD, warp * 8, kk0, lane);
          mma_bf16_16x8x16(qs[0], qs[1], qs[2], qs[3], a0, a1, a2, a3, b0, b1);
        }
        int key = warp * 8 + 2 * tig;
        Ssh[gid * SLD + key] = qs[0];
        Ssh[gid * SLD + key + 1] = qs[1];
        Ssh[(gid + 8) * SLD + key] = qs[2];
        Ssh[(gid + 8) * SLD + key + 1] = qs[3];
      }
      bar_sync(1, NWARP * 32);
      // Parallel online softmax: warp w owns heads [4w, 4w+4), 8 lanes/head,
      // 4 keys per lane; reductions via xor-shuffle within the 8-lane group.
      // Pool-only tiles (col0 >= attend_k, block-uniform) skip softmax and
      // PV entirely: their keys exist for the tap planes alone and must not
      // perturb the attention output or the online-softmax state.
      if (col0 < attend_k) {
        int h = warp * 4 + (lane >> 3);
        int sub = lane & 7;
        float smax = -CUDART_INF_F;
        float sv[BNA / 8];
        #pragma unroll
        for (int j = 0; j < BNA / 8; ++j) {
          int n = sub * (BNA / 8) + j;
          sv[j] = (col0 + n < attend_k && sel_idx[n] <= q_pos)
                      ? Ssh[h * SLD + n] * scale2 : -CUDART_INF_F;
          smax = fmaxf(smax, sv[j]);
        }
        #pragma unroll
        for (int off = 1; off < 8; off <<= 1)
          smax = fmaxf(smax, __shfl_xor_sync(0xffffffffu, smax, off));
        float m_old = m_i[h];
        float m_new = fmaxf(m_old, smax);
        // fix (multi_news F1=0 at L <= topk): a tile whose keys are ALL invalid
        // (-1 incumbent padding slots when L <= topk) leaves m_old == m_new == -inf,
        // so exp2f(-inf - -inf) = exp2f(NaN) = NaN, which poisons l_i/acc (the
        // combine's w=0 cannot undo 0*NaN). Nothing accumulated yet -> corr = 1.
        float corr = (m_new > -CUDART_INF_F) ? exp2f(m_old - m_new) : 1.0f;
        float lsum = 0.0f;
        #pragma unroll
        for (int j = 0; j < BNA / 8; ++j) {
          float p = (sv[j] > -CUDART_INF_F) ? exp2f(sv[j] - m_new) : 0.0f;
          Psh[h * PLD + sub * (BNA / 8) + j] = __float2bfloat16(p);
          lsum += p;
        }
        #pragma unroll
        for (int off = 1; off < 8; off <<= 1)
          lsum += __shfl_xor_sync(0xffffffffu, lsum, off);
        if (sub == 0) {
          m_i[h] = m_new;
          corr_sh[h] = corr;
          l_i[h] = l_i[h] * corr + lsum;
        }
      }
      bar_sync(1, NWARP * 32);
      if (with_select == 5) {
        // Exp-tap (v4_probs): store the raw log2-domain score PER HEAD into
        // SelSlab reinterpreted as SelSlabH [nblk*HB, S, topk+win].  The
        // running m_i at this point is only the max over tiles processed so
        // far — not the split's final max — so every exp/rescale lives in
        // cascade_exp_epilogue_kernel, which reads the FINAL per-split stats
        // from PartM/PartL (numerically identical to rescaling partial
        // exponentials, and simpler).  Acausal/sentinel slots get -inf so
        // the epilogue's exp2 maps them to zero mass.
        for (int i2 = t; i2 < HB * BNA; i2 += NWARP * 32) {
          int h = i2 / BNA, n = i2 % BNA;
          int col = col0 + n;
          if (col >= c_hi) continue;
          float v = (sel_idx[n] <= q_pos) ? Ssh[h * SLD + n] * scale2
                                          : -CUDART_INF_F;
          SelSlab[((long)(hg * HB + h) * S + s_i) * (topk + win) + col] = v;
        }
      } else if (with_select >= 3) {
        // Tap over the incumbent cols: this head-group's weighted 16-head
        // ReLU sum of the raw scores already in Ssh, stored (not accumulated)
        // into this block's plane of SelSlab-as-[nblk, S, topk+win].
        for (int n = t; n < BNA; n += NWARP * 32) {
          int col = col0 + n;
          if (col >= c_hi) continue;
          float acc = 0.0f;
          if (sel_idx[n] <= q_pos) {
            #pragma unroll
            for (int h = 0; h < HB; ++h) {
              float v = Ssh[h * SLD + n];
              acc += tapw_sh[h] * (v > 0.0f ? v : 0.0f);
            }
          }
          SelSlab[((long)hg * S + s_i) * (topk + win) + col] = acc;
        }
      }
      // PV: rescale the register accumulator in-place, then P @ V.
      // Skipped on pool-only tiles (their Psh was never written this
      // iteration and corr_sh is stale from the last attended tile).
      if (col0 < attend_k) {
        float c_lo_f = corr_sh[gid];
        float c_hi_f = corr_sh[gid + 8];
        #pragma unroll
        for (int nt = 0; nt < NT_PV; ++nt) {
          po[nt][0] *= c_lo_f; po[nt][1] *= c_lo_f;
          po[nt][2] *= c_hi_f; po[nt][3] *= c_hi_f;
        }
        #pragma unroll
        for (int kk = 0; kk < BNA / 16; ++kk) {
          unsigned a0, a1, a2, a3;
          ldmatrix_a(a0, a1, a2, a3, Psh, PLD, 0, kk * 16, lane);
          #pragma unroll
          for (int nt = 0; nt < NT_PV; nt += 2) {
            unsigned b[2][2];
            ldmatrix_b_v_x4(b[0][0], b[0][1], b[1][0], b[1][1],
                            Ksh, KLD, kk * 16, warp * DCOL + nt * 8, lane);
            #pragma unroll
            for (int j = 0; j < 2; ++j)
              mma_bf16_16x8x16(po[nt + j][0], po[nt + j][1], po[nt + j][2], po[nt + j][3],
                               a0, a1, a2, a3, b[j][0], b[j][1]);
          }
        }
      }
      bar_sync(1, NWARP * 32);
    }
    if (with_select >= 3 && win > 0) {
      // Admission window: score-only QK tiles over the last `win` positions
      // (this split's share).  Same load + mma path as the main tiles, but
      // no softmax / PV — these keys are selection candidates, not attended
      // keys, so they must not perturb the attention output.
      int prefix = q_pos + 1;
      int w_lo = max(prefix - win, 0);
      int wps = ((prefix - w_lo) + nsplit - 1) / nsplit;
      int wc_lo = w_lo + split * wps;
      int wc_hi = min(wc_lo + wps, prefix);
      int w_tiles = (max(wc_hi - wc_lo, 0) + BNA - 1) / BNA;
      for (int kt = 0; kt < w_tiles; ++kt) {
        int key0 = wc_lo + kt * BNA;
        for (int i = t; i < BNA; i += NWARP * 32) {
          int g = key0 + i;
          sel_idx[i] = (g < wc_hi) ? g : Skv;
        }
        bar_sync(1, NWARP * 32);
        for (int c = t; c < BNA * (DQK / 8); c += NWARP * 32) {
          int key = c / (DQK / 8);
          int chunk = c % (DQK / 8);
          int gidx = sel_idx[key];
          long load_idx = (gidx <= q_pos) ? (long)gidx : 0L;
          __pipeline_memcpy_async((int4*)Ksh + key * (KLD / 8) + chunk,
                                  (const int4*)(KVr + load_idx * DQK) + chunk, sizeof(int4));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        bar_sync(1, NWARP * 32);
        {
          float qs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          for (int kk0 = 0; kk0 < DQK; kk0 += 16) {
            unsigned a0, a1, a2, a3, b0, b1;
            ldmatrix_a(a0, a1, a2, a3, Qsh, KLD, 0, kk0, lane);
            ldmatrix_b_k(b0, b1, Ksh, KLD, warp * 8, kk0, lane);
            mma_bf16_16x8x16(qs[0], qs[1], qs[2], qs[3], a0, a1, a2, a3, b0, b1);
          }
          int key = warp * 8 + 2 * tig;
          Ssh[gid * SLD + key] = qs[0];
          Ssh[gid * SLD + key + 1] = qs[1];
          Ssh[(gid + 8) * SLD + key] = qs[2];
          Ssh[(gid + 8) * SLD + key + 1] = qs[3];
        }
        bar_sync(1, NWARP * 32);
        if (with_select == 5) {
          // Exp-tap: per-head raw log2-domain scores (see the incumbent
          // site).  Window keys are causal by construction (g < wc_hi <=
          // prefix), so no -inf branch is needed here.
          for (int i2 = t; i2 < HB * BNA; i2 += NWARP * 32) {
            int h = i2 / BNA, n = i2 % BNA;
            int g = key0 + n;
            if (g >= wc_hi) continue;
            SelSlab[((long)(hg * HB + h) * S + s_i) * (topk + win)
                    + topk + (g - w_lo)] = Ssh[h * SLD + n] * scale2;
          }
        } else {
          for (int n = t; n < BNA; n += NWARP * 32) {
            int g = key0 + n;
            if (g >= wc_hi) continue;
            float acc = 0.0f;
            #pragma unroll
            for (int h = 0; h < HB; ++h) {
              float v = Ssh[h * SLD + n];
              acc += tapw_sh[h] * (v > 0.0f ? v : 0.0f);
            }
            SelSlab[((long)hg * S + s_i) * (topk + win) + topk + (g - w_lo)] = acc;
          }
        }
        bar_sync(1, NWARP * 32);
      }
    }
    // epilogue: write this block's partials straight from registers
    // (no acc_o smem round-trip; combine kernel reduces the splits).
    // Idle attention halves (split >= nsplit_attn) skip it: their m=-inf
    // partial tells the combine to never read this PartAcc slice.
    if (attn_here) {
      long blk = blockIdx.x;
      #pragma unroll
      for (int nt = 0; nt < NT_PV; ++nt) {
        int col = warp * DCOL + nt * 8 + 2 * tig;
        PartAcc[blk * HB * DV + (long)gid * DV + col] = po[nt][0];
        PartAcc[blk * HB * DV + (long)gid * DV + col + 1] = po[nt][1];
        PartAcc[blk * HB * DV + (long)(gid + 8) * DV + col] = po[nt][2];
        PartAcc[blk * HB * DV + (long)(gid + 8) * DV + col + 1] = po[nt][3];
      }
    }
  } else if (do_sel && (warp - NWARP) < (BNI / 16)) {
    // indexer (fp8 e4m3): this block's disjoint slice of the prefix, all Hidx heads.
    // s[key,head] = IndexK[key,:] . IndexQ[head,:] via mma.m16n8k32.e4m3; the
    // per-row fp8 scales fold into sel_w (q-scale*weight) and IndexKScale (key).
    // Batched rows (idx strides > 0): row s_i's index keys/scales live at
    // their own bases, mirroring the KV kv_stride convention; stride 0 is
    // the legacy single-sequence layout, bit-for-bit unchanged (base + 0).
    const __nv_fp8_e4m3* IndexKr = IndexK + (long)s_i * idx_k_stride;
    const float* IndexKScr = IndexKScale + (long)s_i * idx_scale_stride;
    // Paged index-K (PageTable != nullptr): the serving cache stores keys in
    // 64-token pages laid out [64 x 128 B fp8 keys][64 x fp32 scales]. With
    // BNI == page size, a page-aligned tile IS one page, so the scan walks the
    // page table directly instead of a dense O(L) per-request unpack.
#define IDX_TILE_BASE(k0) ((PageTable != nullptr) \
      ? (IndexK + (long)PageTable[(long)s_i * pt_stride + ((k0) / BNI)] * page_bytes) \
      : (IndexKr + (long)(k0) * DIDX))
    int prefix = q_pos + 1;
    int ips = (prefix + n_slices - 1) / n_slices;
    if (PageTable != nullptr) ips = ((ips + BNI - 1) / BNI) * BNI;  // page-aligned slices
    int g_lo = slice_id * ips;
    int g_hi = min(g_lo + ips, prefix);
    int nt = warp - NWARP;            // 0..3 : this warp's 16-key tile
    int gid = lane >> 2;              // 0..7
    int tid = lane & 3;               // 0..3
    int st = t - NWARP * 32;
    int n_ntiles = Hidx / 8;          // 8-wide head tiles
    int n_tiles = (max(g_hi - g_lo, 0) + BNI - 1) / BNI;
    int mkey0 = nt * 16;

    // Issue an async copy of tile `kt` (full only) into ping-pong buffer `buf`.
#define IDX_PREFETCH(kt, buf) do {                                          \
      int _k0 = g_lo + (kt) * BNI;                                           \
      int4* _dst = (int4*)(IdxKsh + (buf) * (BNI * IKLD));                   \
      const int4* _src = (const int4*)IDX_TILE_BASE(_k0);                    \
      for (int _c = st; _c < BNI * DIDX / 16; _c += (8 - NWARP) * 32) {      \
        int _key = _c / (DIDX / 16), _chunk = _c % (DIDX / 16);             \
        __pipeline_memcpy_async(_dst + _key * (IKLD / 16) + _chunk,         \
                                _src + _c, sizeof(int4));                   \
      }                                                                     \
      __pipeline_commit();                                                  \
    } while (0)

    int first_full = (n_tiles > 0 && (g_lo + BNI) <= skv_row);
    if (first_full) IDX_PREFETCH(0, 0);
    for (int kt = 0; kt < n_tiles; ++kt) {
      int buf = kt & 1;
      int k0 = g_lo + kt * BNI;
      const __nv_fp8_e4m3* tb = IDX_TILE_BASE(k0);
      __nv_fp8_e4m3* Kbuf = IdxKsh + buf * (BNI * IKLD);
      if (k0 + BNI <= skv_row) {
        __pipeline_wait_prior(0);             // this tile's prefetch lands
      } else {
        for (int i = st; i < BNI * DIDX; i += (8 - NWARP) * 32) {
          int key = i / DIDX, d = i % DIDX, g = k0 + key;
          Kbuf[key * IKLD + d] = (g < skv_row) ? tb[(long)key * DIDX + d] : __nv_fp8_e4m3(0.0f);
        }
      }
      bar_sync(2, (8 - NWARP) * 32);
      // Prefetch the next full tile so its load hides behind this tile's mma+reduce.
      if (kt + 1 < n_tiles && (g_lo + (kt + 1) * BNI + BNI) <= skv_row)
        IDX_PREFETCH(kt + 1, (kt + 1) & 1);
      // fp8 GEMM on native f16 HMMA (exact e4m3 -> f16), head reduction in
      // registers: the same helpers as hide_paged_kernel, so paged == dense
      // stays bit-identical (gate G1).
      uint2 a[DIDX / 32][4];
      #pragma unroll
      for (int kk = 0; kk < DIDX / 32; ++kk) {
        const __nv_fp8_e4m3* Ab = Kbuf + (mkey0 + gid) * IKLD + kk * 32 + tid * 4;
        a[kk][0] = hp_e4m3x4_to_f16x4(*(const unsigned int*)(Ab));
        a[kk][1] = hp_e4m3x4_to_f16x4(*(const unsigned int*)(Ab + 8 * IKLD));
        a[kk][2] = hp_e4m3x4_to_f16x4(*(const unsigned int*)(Ab + 16));
        a[kk][3] = hp_e4m3x4_to_f16x4(*(const unsigned int*)(Ab + 8 * IKLD + 16));
      }
      float accA = 0.0f, accB = 0.0f;   // keys mkey0+gid, mkey0+gid+8
      #pragma unroll 4
      for (int ntile = 0; ntile < n_ntiles; ++ntile) {
        int n0 = ntile * 8;
        float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
        #pragma unroll
        for (int kk = 0; kk < DIDX / 32; ++kk) {
          const __nv_fp8_e4m3* Bb = IdxQsh + (n0 + gid) * IKLD + kk * 32 + tid * 4;
          uint2 b0 = hp_e4m3x4_to_f16x4(*(const unsigned int*)(Bb));
          uint2 b1 = hp_e4m3x4_to_f16x4(*(const unsigned int*)(Bb + 16));
          hp_index_mma_k32(c0, c1, c2, c3, a[kk][0], a[kk][1], a[kk][2], a[kk][3], b0, b1);
        }
        hp_index_reduce_ntile(accA, accB, c0, c1, c2, c3, sel_w[n0 + 2 * tid], sel_w[n0 + 2 * tid + 1]);
      }
      accA = hp_index_reduce_lanes(accA);
      accB = hp_index_reduce_lanes(accB);
      if (tid == 0) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
          int n = mkey0 + gid + half * 8;
          int g = k0 + n;
          if (g >= g_hi || g >= skv_row) continue;
          float acc = 0.0f;
          if (g <= q_pos) {
            acc = half ? accB : accA;
            // per-key fp8 scale: page-resident (fp32 block after the 64 keys) or dense
            acc *= (PageTable != nullptr) ? ((const float*)(tb + BNI * DIDX))[n] : IndexKScr[g];
            // shared-primitive epilogue: bucket the final logit while it is
            // still in a register, so tl_topk can skip its histogram pass
            if (with_select == 2) atomicAdd(&s_hist[sel_bucket16(acc)], 1);
          }
          SelSlab[(long)s_i * slab_row + g] = acc;
        }
      }
    }
#undef IDX_PREFETCH
#undef IDX_TILE_BASE
    if (with_select == 2) {
      // per-slice partial histograms merge into the per-query histogram
      bar_sync(2, (8 - NWARP) * 32);
      for (int i = st; i < 256; i += (8 - NWARP) * 32)
        if (s_hist[i] != 0) atomicAdd(&SelHist[(long)s_i * 256 + i], s_hist[i]);
    }
  }
  __syncthreads();

  // m_i is tracked in log2 domain (exp2f softmax); the combine kernel reduces
  // splits with natural expf, so convert the stored max: m_nat = m_log2 * ln2.
  // (l and the accumulator are domain-invariant: exp2(x2-m2) == exp(x-m).)
  long blk2 = blockIdx.x;
  for (int i = t; i < HB; i += blockDim.x) {
    PartM[blk2 * HB + i] = m_i[i] * 0.6931471805599453f;
    PartL[blk2 * HB + i] = l_i[i];
  }

  if (with_select == 4) {
    // Folded cascade selection (threadfence-reduction pattern): every CTA
    // fences its tap-plane writes, then signals arrival on the per-row
    // counter; the last of the row's nblk*nsplit CTAs sees all planes and
    // runs the radix select inline. The attention smem is dead by now, so
    // the select buffers (~19.5 KB) are carved from its head; they end well
    // below m_i/l_i (~58 KB in), whose reads completed before this point.
    __threadfence();
    __shared__ int s_elect;
    if (t == 0) {
      int prev = atomicAdd(&SelCounter[s_i], 1);
      s_elect = (prev == nblk * nsplit - 1) ? 1 : 0;
    }
    __syncthreads();
    if (s_elect) {
      float* f_vals = (float*)smem;
      int* f_hist = (int*)(f_vals + SELN);
      int* f_subhist = f_hist + 256;
      int* f_tie = f_subhist + 256;
      int* f_scal = f_tie + 256;
      cascade_select_row(SelSlab, Indices, NextIdx, s_i, S, nblk, topk, win,
                         Skv, f_vals, f_hist, f_subhist, f_tie, f_scal);
    }
  }
}

// Cascade selection epilogue: one block per query row. Sums the per-head-group
// tap planes into the final union logits and marks window slots whose key is
// already in the incumbent set with -inf (exact dedupe: duplicates carry
// identical scores, so dropping the window copy is lossless). The subsequent
// top-k over the union is then exact and single-launch.
extern "C" __global__ void cascade_epilogue_kernel(
    const float* __restrict__ Planes,  // [nblk, S, topk+win]
    const int* __restrict__ Indices,   // [S, topk]
    float* __restrict__ TapOut,        // [S, topk+win]
    int S, int nblk, int topk, int win, int Skv, int batched) {
  int s = blockIdx.x;
  int ntap = topk + win;
  int t = threadIdx.x;
  // batched: every row is an independent sequence decoding at position
  // Skv-1 (matches the main kernel's kv_stride>0 semantics); legacy: the S
  // rows are the trailing S positions of one sequence.
  int q_pos = batched ? (Skv - 1) : (Skv - S) + s;
  int prefix = q_pos + 1;
  int w_lo = max(prefix - win, 0);
  int w_span = prefix - w_lo;   // window slots beyond this were never written
  for (int c = t; c < ntap; c += blockDim.x) {
    float acc = -CUDART_INF_F;
    if (c < topk + w_span) {
      acc = 0.0f;
      for (int p = 0; p < nblk; ++p) acc += Planes[((long)p * S + s) * ntap + c];
    }
    TapOut[(long)s * ntap + c] = acc;
  }
  __syncthreads();
  for (int c = t; c < topk; c += blockDim.x) {
    int g = Indices[(long)s * topk + c];
    if (g > q_pos) TapOut[(long)s * ntap + c] = -CUDART_INF_F;
    else if (g >= w_lo)
      TapOut[(long)s * ntap + topk + (g - w_lo)] = -CUDART_INF_F;
  }
}

// Exp-tap epilogue (v4_probs): turns the mode-5 per-head raw-score planes
// into union logits  sum_hg sum_h exp2(s2 - m2_g[h]) / Z2_g[h]  with per-head
// (m, Z) over the ATTEND slice — reduced from the PartM/PartL split
// statistics the main kernel already writes.  PartM is stored in the
// NATURAL-log domain (the kernel epilogue multiplies by ln2 for the combine
// kernel), while the planes are log2-domain; exp2(s2 - m_nat*log2e) ==
// exp(s_nat - m_nat), so one domain conversion of the max suffices, and
// PartL is domain-invariant.  Because the planes hold RAW s2 (not partial
// exponentials), no per-column split mapping is needed: exp2 against the
// GLOBAL per-head max is numerically direct.  Window-dedupe and sentinel
// handling mirror cascade_epilogue_kernel.  One block per query row,
// blockDim 256.  Row semantics mirror the main kernel's kv_stride switch:
// batched != 0 means every row is an independent sequence decoding at
// position Skv-1 (pairs with the mode-5 kv_stride>0 launch); legacy (0)
// keeps the trailing-S-positions-of-one-sequence semantics.
#define EPI_MAXH 256
extern "C" __global__ void cascade_exp_epilogue_kernel(
    const float* __restrict__ PlanesH,  // [nblk*HB, S, topk+win] raw log2 s2
    const float* __restrict__ PartM,    // [S*nblk*nsplit, HB] natural-log max
    const float* __restrict__ PartL,    // [S*nblk*nsplit, HB] denominators
    const int* __restrict__ Indices,    // [S, topk]
    float* __restrict__ TapOut,         // [S, topk+win]
    float* __restrict__ ExpStats,       // [S, 2, nblk*HB]: m2_g (log2) | 1/Z
    int S, int nblk, int topk, int win, int Skv, int nsplit, int batched) {
  int s = blockIdx.x;
  int t = threadIdx.x;
  int nh = nblk * HB;
  int ntap = topk + win;
  int q_pos = batched ? (Skv - 1) : (Skv - S) + s;
  int prefix = q_pos + 1;
  int w_lo = max(prefix - win, 0);
  int w_span = prefix - w_lo;   // window slots beyond this were never written
  __shared__ float m2g[EPI_MAXH];    // per-head global attend max, log2 domain
  __shared__ float invz[EPI_MAXH];   // per-head 1/Z (0 = no attended mass)
  for (int i = t; i < nh; i += blockDim.x) {
    int hg = i / HB, h = i % HB;
    float m = -CUDART_INF_F;
    for (int sp = 0; sp < nsplit; ++sp) {
      long blk = ((long)s * nblk + hg) * nsplit + sp;
      m = fmaxf(m, PartM[blk * HB + h]);
    }
    float z = 0.0f;
    if (m > -CUDART_INF_F) {
      for (int sp = 0; sp < nsplit; ++sp) {
        long blk = ((long)s * nblk + hg) * nsplit + sp;
        float ms = PartM[blk * HB + h];
        if (ms > -CUDART_INF_F) z += PartL[blk * HB + h] * __expf(ms - m);
      }
    }
    m2g[i] = m * 1.44269504f;
    invz[i] = (z > 0.0f) ? (1.0f / z) : 0.0f;
    // Export the per-head global stats so the host can score OUT-OF-PLANE
    // candidates (the sink positions) with the exact same normalization the
    // plane columns receive — required for a commensurate union top-k.
    ExpStats[((long)s * 2 + 0) * nh + i] = m2g[i];
    ExpStats[((long)s * 2 + 1) * nh + i] = invz[i];
  }
  __syncthreads();
  for (int c = t; c < ntap; c += blockDim.x) {
    float acc = -CUDART_INF_F;
    if (c < topk + w_span) {
      acc = 0.0f;
      for (int i = 0; i < nh; ++i) {
        // invz gate: a head with no attended mass contributes nothing, and
        // the gate guards the inf that exp2(s2 - (-inf)) would produce.
        // Acausal plane slots hold -inf, so their exp2 is exactly 0.
        if (invz[i] > 0.0f)
          acc += exp2f(PlanesH[((long)i * S + s) * ntap + c] - m2g[i]) * invz[i];
      }
    }
    TapOut[(long)s * ntap + c] = acc;
  }
  __syncthreads();
  for (int c = t; c < topk; c += blockDim.x) {
    int g = Indices[(long)s * topk + c];
    if (g > q_pos) TapOut[(long)s * ntap + c] = -CUDART_INF_F;
    else if (g >= w_lo)
      TapOut[(long)s * ntap + topk + (g - w_lo)] = -CUDART_INF_F;
  }
}

// Single-launch cascade selection: plane-sum + dedupe + top-k + map to
// absolute key indices, one block per query row. Selection is a three-level
// radix threshold (histogram over a monotone 32-bit float key: one byte per
// level, 24 bits total); ties beyond 24 value bits resolve deterministically
// by smallest slot, so the selected SET is reproducible run to run. Unlike a sort,
// every pass is O(ntap) with 256-way parallelism, so a single block
// finishes in microseconds (the in-block bitonic sort this replaces was
// latency-bound at ~0.12 ms for B=1).

__device__ __forceinline__ unsigned key32(float x) {
  unsigned u = __float_as_uint(x);
  return (x < 0.0f) ? ~u : (u | 0x80000000u);
}

// Body shared between the standalone kernel and the in-fused-kernel fold
// (with_select==4): all shared-memory buffers are caller-provided so the fold
// can carve them from the fused kernel's dynamic smem. `scal` holds the eight
// cross-thread scalars: need, bin, sub_need, sub_bin, sub3_need, sub3_bin,
// out_cnt, tie_list_cnt. Every thread of the calling block must enter (the
// body uses __syncthreads()), and blockDim must be 256.
__device__ void cascade_select_row(
    const float* __restrict__ Planes,  // [nblk, S, topk+win]
    const int* __restrict__ Indices,   // [S, topk]
    int* __restrict__ NextIdx,         // [S, topk] absolute keys (-1 pad)
    int s, int S, int nblk, int topk, int win, int Skv,
    float* vals, int* hist, int* subhist, int* tie_slots, int* scal) {
  int& s_need = scal[0];
  int& s_bin = scal[1];
  int& s_sub_need = scal[2];
  int& s_sub_bin = scal[3];
  int& s_sub3_need = scal[4];
  int& s_sub3_bin = scal[5];
  int& out_cnt = scal[6];
  int& tie_list_cnt = scal[7];
  int ntap = topk + win;
  int t = threadIdx.x;
  int q_pos = (Skv - S) + s;
  int prefix = q_pos + 1;
  int w_lo = max(prefix - win, 0);
  int w_span = prefix - w_lo;   // window slots beyond this were never written

  for (int c = t; c < 256; c += blockDim.x) { hist[c] = 0; subhist[c] = 0; }
  if (t == 0) { out_cnt = 0; tie_list_cnt = 0; }
  for (int c = t; c < ntap; c += blockDim.x) {
    float acc = -CUDART_INF_F;
    if (c < topk + w_span) {
      acc = 0.0f;
      for (int p = 0; p < nblk; ++p) acc += Planes[((long)p * S + s) * ntap + c];
    }
    vals[c] = acc;
  }
  __syncthreads();
  // Dedupe + sentinel guard: a window slot whose key is in the incumbent set
  // is dropped (identical score by construction, lossless), and incumbent
  // slots holding acausal/padded sentinel indices are never selectable.
  for (int c = t; c < topk; c += blockDim.x) {
    int g = Indices[(long)s * topk + c];
    if (g > q_pos) vals[c] = -CUDART_INF_F;
    else if (g >= w_lo) vals[topk + (g - w_lo)] = -CUDART_INF_F;
  }
  __syncthreads();
  // Level 1: histogram of the key's top byte over finite candidates.
  for (int c = t; c < ntap; c += blockDim.x)
    if (isfinite(vals[c])) atomicAdd(&hist[key32(vals[c]) >> 24], 1);
  __syncthreads();
  if (t == 0) {
    int above = 0, b = 255;
    for (; b >= 0; --b) {
      if (above + hist[b] >= topk || b == 0) break;
      above += hist[b];
    }
    s_bin = b;
    s_need = topk - above;   // taken from bin b (may exceed hist[b] if short)
  }
  __syncthreads();
  int bin = s_bin;
  // Level 2: histogram of the second byte within the threshold bin.
  for (int c = t; c < ntap; c += blockDim.x) {
    if (!isfinite(vals[c])) continue;
    unsigned k = key32(vals[c]);
    if ((int)(k >> 24) == bin) atomicAdd(&subhist[(k >> 16) & 0xFF], 1);
  }
  __syncthreads();
  if (t == 0) {
    int above = 0, b = 255;
    for (; b >= 0; --b) {
      if (above + subhist[b] >= s_need || b == 0) break;
      above += subhist[b];
    }
    s_sub_bin = b;
    s_sub_need = s_need - above;
  }
  __syncthreads();
  int sub_bin = s_sub_bin;
  // Level 3: histogram of the third byte within the level-2 bucket (24-bit
  // total refinement), reusing hist[] which level 1 no longer needs.
  for (int c = t; c < 256; c += blockDim.x) hist[c] = 0;
  __syncthreads();
  for (int c = t; c < ntap; c += blockDim.x) {
    if (!isfinite(vals[c])) continue;
    unsigned k = key32(vals[c]);
    if ((int)(k >> 24) == bin && (int)((k >> 16) & 0xFF) == sub_bin)
      atomicAdd(&hist[(k >> 8) & 0xFF], 1);
  }
  __syncthreads();
  if (t == 0) {
    int above = 0, b = 255;
    for (; b >= 0; --b) {
      if (above + hist[b] >= s_sub_need || b == 0) break;
      above += hist[b];
    }
    s_sub3_bin = b;
    s_sub3_need = s_sub_need - above;   // from the final 24-bit tie set
  }
  __syncthreads();
  int sub3_bin = s_sub3_bin;
  // Gather the final 24-bit tie set; take the s_sub3_need smallest SLOTS so
  // the selected SET is deterministic (ties beyond 24 value bits resolve by
  // slot order, which is top-k set semantics with a fixed tie-break).
  for (int c = t; c < ntap; c += blockDim.x) {
    if (!isfinite(vals[c])) continue;
    unsigned k = key32(vals[c]);
    if ((int)(k >> 24) == bin && (int)((k >> 16) & 0xFF) == sub_bin
        && (int)((k >> 8) & 0xFF) == sub3_bin) {
      int pos = atomicAdd(&tie_list_cnt, 1);
      if (pos < 256) tie_slots[pos] = c;
    }
  }
  __syncthreads();
  // With continuous scores the 24-bit tie set is tiny; if it ever exceeds
  // the buffer, fail loudly rather than select approximately.
  if (t == 0 && tie_list_cnt > 256) {
    // Poison the output so the host-side set-agreement gate trips.
    NextIdx[(long)s * topk] = -2;
  }
  // Serial selection of the smallest slots within the (tiny) tie set.
  if (t == 0) {
    int take_n = min(s_sub3_need, min(tie_list_cnt, 256));
    for (int i = 0; i < take_n; ++i) {
      int best = -1, best_j = -1;
      for (int j = 0; j < min(tie_list_cnt, 256); ++j) {
        int sl = tie_slots[j];
        if (sl >= 0 && (best == -1 || sl < best)) { best = sl; best_j = j; }
      }
      tie_slots[best_j] = -1;
      int out = (best < topk) ? Indices[(long)s * topk + best]
                              : (w_lo + (best - topk));
      int pos = atomicAdd(&out_cnt, 1);
      if (pos < topk) NextIdx[(long)s * topk + pos] = out;
    }
  }
  __syncthreads();
  // Collect everything strictly above the final 24-bit threshold bucket.
  for (int c = t; c < ntap; c += blockDim.x) {
    if (!isfinite(vals[c])) continue;
    unsigned k = key32(vals[c]);
    int b1 = (int)(k >> 24);
    if (b1 < bin) continue;
    bool take = false;
    if (b1 > bin) take = true;
    else {
      int b2 = (int)((k >> 16) & 0xFF);
      if (b2 > sub_bin) take = true;
      else if (b2 == sub_bin && (int)((k >> 8) & 0xFF) > sub3_bin) take = true;
    }
    if (take) {
      int pos = atomicAdd(&out_cnt, 1);
      if (pos < topk) {
        int out = (c < topk) ? Indices[(long)s * topk + c] : (w_lo + (c - topk));
        NextIdx[(long)s * topk + pos] = out;
      }
    }
  }
  __syncthreads();
  // Pad when fewer than topk finite candidates exist.
  for (int c = out_cnt + t; c < topk; c += blockDim.x)
    NextIdx[(long)s * topk + c] = -1;
}

extern "C" __global__ void cascade_select_kernel(
    const float* __restrict__ Planes,  // [nblk, S, topk+win]
    const int* __restrict__ Indices,   // [S, topk]
    int* __restrict__ NextIdx,         // [S, topk] absolute keys (-1 pad)
    int S, int nblk, int topk, int win, int Skv) {
  __shared__ float vals[SELN];
  __shared__ int hist[256];
  __shared__ int subhist[256];
  __shared__ int tie_slots[256];   // final 24-bit-tie members, slot-ordered take
  __shared__ int scal[8];
  cascade_select_row(Planes, Indices, NextIdx, blockIdx.x, S, nblk, topk, win,
                     Skv, vals, hist, subhist, tie_slots, scal);
}

extern "C" __global__ void combine_kernel(
    const float* __restrict__ PartAcc, const float* __restrict__ PartM,
    const float* __restrict__ PartL, __nv_bfloat16* __restrict__ Out,
    int S, int H, int nsplit, int nsplit_used) {
  // nsplit = partial-buffer stride per (row, head-group); nsplit_used = how many
  // leading splits carried attention (the rest hold neutral m=-inf partials and
  // are skipped entirely: this loop is per-output serial, so it must not scale
  // with the indexer's split count).
  int nblk = H / HB;
  int bid = blockIdx.x;
  int hg = bid % nblk;
  int s_i = bid / nblk;
  int t = threadIdx.x;
  __shared__ float gm[HB];
  __shared__ float gl[HB];
  if (t < HB) {
    int h = t;
    float m = -CUDART_INF_F;
    for (int sp = 0; sp < nsplit_used; ++sp) {
      long blk = (long)bid * nsplit + sp;
      m = fmaxf(m, PartM[blk * HB + h]);
    }
    float l = 0.0f;
    if (m > -CUDART_INF_F) {
      for (int sp = 0; sp < nsplit_used; ++sp) {
        long blk = (long)bid * nsplit + sp;
        float ms = PartM[blk * HB + h];
        if (ms > -CUDART_INF_F) l += PartL[blk * HB + h] * __expf(ms - m);
      }
    }
    gm[h] = m; gl[h] = l;
  }
  __syncthreads();
  for (int i = t; i < HB * DV; i += blockDim.x) {
    int h = i / DV, d = i % DV;
    float m = gm[h];
    float out = 0.0f;
    if (m > -CUDART_INF_F) {
      float acc = 0.0f;
      for (int sp = 0; sp < nsplit_used; ++sp) {
        long blk = (long)bid * nsplit + sp;
        float ms = PartM[blk * HB + h];
        if (ms > -CUDART_INF_F) acc += PartAcc[(blk * HB + h) * DV + d] * __expf(ms - m);
      }
      float l = gl[h];
      out = acc / (l > 0.0f ? l : 1.0f);
    }
    Out[(((long)s_i * H) + hg * HB + h) * DV + d] = __float2bfloat16(out);
  }
}

// ===========================================================================
// HIDE paged kernel (serving path of launch_fused_sparse_splitk_paged).
//
// Blocks are DECOUPLED by role instead of sharing attention+indexer warps:
//   split <  nsa : ATTENTION block. 8 warps = 2 groups x 4 warps. Group g
//                  owns the BNA-key tiles kt == g (mod 2) of this split's
//                  slice of the top-k keys, with its own K buffer, online-
//                  softmax state and named barrier, so one group's KV gather
//                  latency hides behind the other group's QK/softmax/PV. Each
//                  group writes its own partial slot (2 per block); the LAST
//                  arriving attention block of (row, head-group) reduces all
//                  2*nsa slots into Out (threadfence election on a self-
//                  resetting counter) -- no separate combine launch.
//   split >= nsa : INDEXER block. 8 warps x 16 keys = 128-key (2-page) tiles
//                  of this block's page-aligned prefix slice, HP_ISTAGES-deep
//                  cp.async prefetch, per-key logits straight to SelSlab. The
//                  row's slab tail [prefix, slab_row) is written -inf here, so
//                  the host needs no fill.
// Per-key indexer arithmetic (fp8 mma k-order, sequential head reduction,
// page-resident scale) is IDENTICAL to fused_sparse_splitk_kernel, so the
// paged logits stay bit-exact vs the dense entry (gate G1).
// ===========================================================================
#define HP_BNI 128            // indexer keys per tile (2 pages)
#define HP_ISTAGES 4          // indexer prefetch depth (tiles in flight)
#define HP_ISLD (Hidx + 1)    // odd row stride -> conflict-free key-major reads
#define HP_KSPLIT (DQK / 16 / NWARP)   // 9 k-steps of the QK dot per warp (k-split QK)
#define HP_GSTATE ((2 * 16 * SLD) * 4 + (16 * PLD) * 2 + 3 * 16 * 4 + 2 * BNA * 4)  // per-group state bytes
#define HP_MAX_NSA 32         // combine weight table [slot][head]

// Phase-timing instrumentation (diagnostics only): when hp_dbg_on != 0, virtual
// block v (< HP_DBG_BLOCKS) stamps globaltimer (ns) at fixed phase ids; read
// back via hide_debug_read(). Off by default: one predicated load per stamp.
#define HP_DBG_BLOCKS 512
#define HP_DBG_PH 16
__device__ int hp_dbg_on = 0;
// scan diagnostics (gate only): bit0 = skip the indexer MMA (memory floor),
// bit1 = skip the index-K prefetch (compute floor on stale smem)
__device__ int hp_diag_mode = 0;
__device__ unsigned long long hp_dbg[HP_DBG_BLOCKS * HP_DBG_PH];
__device__ __forceinline__ unsigned long long hp_now() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
#define HP_STAMP(v, ph) do { if (hp_dbg_on && threadIdx.x == 0 && (v) < HP_DBG_BLOCKS) \
    hp_dbg[(v) * HP_DBG_PH + (ph)] = hp_now(); } while (0)

#define HP_NCOMB 8            // combine blocks per (row, head-group): 64 output cols each
// Counter slots inside AttnCounter ([S, 256] int32, zero at first launch, self-resetting):
//   [s*256 + hg]       arrivals of the row/head-group's attention blocks
//   [s*256 + 16 + hg]  finished combine blocks (the last one re-arms both)

// ---------------------------------------------------------------- v10 additions
// (1) fp8 latent-KV pool consumed DIRECTLY by the attention blocks (serving's
//     656 B rows: 512 e4m3 nope | 4 fp32 group scales | 64 bf16 rope). Every
//     attended row of the block is gathered up front into registers, dequantized
//     exactly as sglang's dequantize_k_cache does (fp8 -> f32, * scale, RN -> bf16)
//     and written into the group's bf16 K tile, so the wrapper's index_select +
//     dequant launch disappears and the whole gather is one latency instead of
//     one per tile. Requires topk == 2 * BNA * nsplit_attn (one tile per group).
// (2) in-kernel top-k SELECTION over the full-prefix indexer logits, mirroring
//     the production topk_v2 algorithm (10-bit fp16 coarse histogram -> threshold
//     bin -> elements above the bin emitted directly, elements in the bin resolved
//     by an exact radix select over <= HP_MAXTIE candidates). The row's indexer
//     blocks cooperate through a global histogram and two arrival counters (all
//     of a row's indexer blocks are co-resident: checked by the launcher against
//     the occupancy-derived capacity, same in-order-dispatch assumption as the
//     combine blocks); the LAST arriving block resolves the ties, writes NextIdx
//     in KV-slot space through the page-64 table, and re-arms the row's scratch.
#define KV_FP8_ROW 656
#define HP_HBITS 10
#define HP_HSIZE (1 << HP_HBITS)
#define HP_MAXTIE 2048
#define HP_MAX_TILES 128            // indexer tiles per block (page ids cached in smem)
#define HP_LG_KEYS 2048             // slice logits kept in smem for the select (else re-read from L2)
// per-row select scratch (uint32 words): [0,1024) histogram, 1024 arrive, 1025 count_gt,
// 1026 count_eq, 1027 done, [1056, 1056+2*HP_MAXTIE) tie (key, idx) pairs
#define HP_SEL_ARRIVE 1024
#define HP_SEL_CNT_GT 1025
#define HP_SEL_CNT_EQ 1026
#define HP_SEL_DONE 1027
#define HP_SEL_TIE 1056
#define HP_SEL_CAND (HP_SEL_TIE + 2 * HP_MAXTIE)   // approx: n_idx * m (key, idx) pairs
#define HP_MAXCAND 8192                            // n_idx * m bound (merge stages them in smem)
#define HP_SEL_STRIDE (HP_SEL_CAND + 2 * HP_MAXCAND)

// order-preserving fp32 -> u32 key (production extract_exact_bin)
__device__ __forceinline__ unsigned hp_exact_key(float x) {
  unsigned b = __float_as_uint(x);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
// production extract_coarse_bin<10>: fp16 (RN) ordered key, top 10 bits
__device__ __forceinline__ unsigned hp_coarse_bin(float x) {
  unsigned short bits = __half_as_ushort(__float2half_rn(x));
  unsigned short key = (bits & 0x8000) ? (unsigned short)(~bits) : (unsigned short)(bits | 0x8000);
  return ((unsigned)key) >> (16 - HP_HBITS);
}
__device__ __forceinline__ int hp_page_to_slot(const int* __restrict__ PT, int g) {
  return (PT[g >> 6] << 6) | (g & 63);
}
__device__ __forceinline__ unsigned hp_ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
// ---- Hopper wgmma fp8 scan path -------------------------------------------
// K-major operands in the 128B-swizzle canonical layout: row r (128 B = one
// key's / head's 128 e4m3 dims) at r*128, 16-B chunk c stored at chunk c ^ (r & 7);
// 8-row atoms of 1 KB (SBO = 1024), atom base 1024-B aligned.
#define HP_WG_STAGE 16384      // 128 keys x 128 B
#define HP_WG_QBYTES 8192      // <= 64 heads x 128 B
__device__ __forceinline__ unsigned long long hp_wg_desc(unsigned smem_addr) {
  unsigned long long d = 0;
  d |= (unsigned long long)((smem_addr >> 4) & 0x3fffu);          // start address
  d |= (unsigned long long)(1u) << 16;                              // LBO (unused for swizzled K-major)
  d |= (unsigned long long)((1024u >> 4) & 0x3fffu) << 32;          // SBO: 8-row atom stride
  d |= (unsigned long long)(1u) << 62;                              // 128B swizzle
  return d;
}
__device__ __forceinline__ void hp_wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void hp_wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void hp_wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }
// D[64 keys x 64 heads] (+)= A[64 x 32 e4m3] * B[64 x 32 e4m3]^T ; d[4j..4j+3] = (row gid, cols 8j+2tid..+1), (row gid+8, same)
__device__ __forceinline__ void hp_wgmma_64x64x32(float* d, unsigned long long da, unsigned long long db, int scale_d) {
  asm volatile(
    "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
    "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
    "%32, %33, p, 1, 1;\n}\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
      "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
      "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
      "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
    : "l"(da), "l"(db), "r"(scale_d));
}

// exact e4m3 -> bf16 dequant of 4 packed fp8 (one 32-bit word) with one fp32 scale:
// fp8 -> f16 (exact) -> f32 (exact) -> * scale (f32) -> bf16 RN, the arithmetic of
// sglang's _dequantize_k_cache_fast_kernel, so the tile equals the wrapper's path.
__device__ __forceinline__ uint2 hp_dequant4(unsigned w, float s) {
  uint2 h = hp_e4m3x4_to_f16x4(w);
  __half2 h01 = *reinterpret_cast<const __half2*>(&h.x);
  __half2 h23 = *reinterpret_cast<const __half2*>(&h.y);
  float2 f01 = __half22float2(h01);
  float2 f23 = __half22float2(h23);
  __nv_bfloat162 b01 = __floats2bfloat162_rn(f01.x * s, f01.y * s);
  __nv_bfloat162 b23 = __floats2bfloat162_rn(f23.x * s, f23.y * s);
  uint2 r;
  r.x = *reinterpret_cast<const unsigned*>(&b01);
  r.y = *reinterpret_cast<const unsigned*>(&b23);
  return r;
}

// Warp-aggregated shared-memory updates: contended smem atomics serialize per
// lane (32-way when a warp's lanes share an address -- the common case for a
// radix histogram whose top bytes are nearly constant), so bins are counted by
// __match_any_sync peers and cursors allocated by one ballot per warp.
__device__ __forceinline__ void hp_warp_hist_add(unsigned* hist, unsigned bin, bool pred) {
  unsigned act = __ballot_sync(0xffffffffu, pred);
  if (pred) {
    unsigned peers = __match_any_sync(act, bin);
    if ((int)(threadIdx.x & 31) == __ffs(peers) - 1) atomicAdd(&hist[bin], (unsigned)__popc(peers));
  }
}
__device__ __forceinline__ unsigned hp_warp_alloc(unsigned* ctr, bool pred) {
  const unsigned lane = threadIdx.x & 31;
  unsigned m = __ballot_sync(0xffffffffu, pred);
  unsigned base = 0;
  if (m) {
    int leader = __ffs(m) - 1;
    if ((int)lane == leader) base = atomicAdd(ctr, (unsigned)__popc(m));
    base = __shfl_sync(0xffffffffu, base, leader);
  }
  return base + (unsigned)__popc(m & ((1u << lane) - 1u));
}

// Block-wide sum of one value per thread (8 warps). `red` is a 16-word smem
// scratch double-buffered by `par` so back-to-back calls need no extra barrier.
__device__ __forceinline__ unsigned hp_block_sum(unsigned v, unsigned* red, int par) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  if ((threadIdx.x & 31) == 0) red[par * 8 + (threadIdx.x >> 5)] = v;
  __syncthreads();
  unsigned s = 0;
  #pragma unroll
  for (int w = 0; w < 8; ++w) s += red[par * 8 + w];
  return s;
}
// Block-wide exclusive prefix of one value per thread; returns this thread's
// offset and writes the block total to *total. `red` as above.
__device__ __forceinline__ unsigned hp_block_excl_scan(unsigned v, unsigned* red, int par, unsigned* total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  unsigned inc = v;
  #pragma unroll
  for (int off = 1; off < 32; off <<= 1) { unsigned n = __shfl_up_sync(0xffffffffu, inc, off); if (lane >= off) inc += n; }
  if (lane == 31) red[par * 8 + warp] = inc;
  __syncthreads();
  unsigned before = 0, tot = 0;
  #pragma unroll
  for (int w = 0; w < 8; ++w) { unsigned x = red[par * 8 + w]; before += (w < warp) ? x : 0u; tot += x; }
  *total = tot;
  return before + inc - v;
}
// Exact top-`remain` select over `n_per` register-resident keys per thread
// (candidate c = t + 256*j has key[j]; inactive keys are 0 = smallest). Finds
// T = max{x : #(key >= x) >= remain} by bisection over the ordered-key space
// (count = 32 compares + one block reduction per step; steps <= bits of the
// [min, max] span), then emits deterministic positions: keys > T in block-scan
// order, then keys == T until the remaining slots are filled. `emit(pos, j)` is
// called for this thread's key j at output position pos. No atomics.
template <int NPER, typename Emit>
__device__ __forceinline__ void hp_bisect_select(const unsigned* key, int remain, unsigned* red, Emit emit) {
  unsigned kmin = 0xffffffffu, kmax = 0u;
  #pragma unroll
  for (int j = 0; j < NPER; ++j) { kmin = min(kmin, key[j]); kmax = max(kmax, key[j]); }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    kmin = min(kmin, __shfl_xor_sync(0xffffffffu, kmin, off));
    kmax = max(kmax, __shfl_xor_sync(0xffffffffu, kmax, off));
  }
  if ((threadIdx.x & 31) == 0) { red[16 + (threadIdx.x >> 5)] = kmin; red[24 + (threadIdx.x >> 5)] = kmax; }
  __syncthreads();
  kmin = 0xffffffffu; kmax = 0u;
  #pragma unroll
  for (int w = 0; w < 8; ++w) { kmin = min(kmin, red[16 + w]); kmax = max(kmax, red[24 + w]); }
  unsigned lo = kmin, hi = kmax;   // #(key >= lo) = all >= remain (caller guarantees)
  int par = 0;
  #pragma unroll 1
  while (lo < hi) {
    unsigned mid = lo + ((hi - lo) >> 1) + 1u;
    unsigned c = 0;
    #pragma unroll
    for (int j = 0; j < NPER; ++j) c += (key[j] >= mid);
    c = hp_block_sum(c, red, par); par ^= 1;
    if (c >= (unsigned)remain) lo = mid; else hi = mid - 1u;
  }
  const unsigned T = lo;
  unsigned n_gt = 0, n_eq = 0;
  #pragma unroll
  for (int j = 0; j < NPER; ++j) { n_gt += (key[j] > T); n_eq += (key[j] == T); }
  unsigned tot_gt, tot_eq;
  unsigned base_gt = hp_block_excl_scan(n_gt, red, par, &tot_gt); par ^= 1;
  unsigned base_eq = hp_block_excl_scan(n_eq, red, par, &tot_eq); par ^= 1;
  const unsigned eq_slots = (unsigned)remain - tot_gt;   // tot_gt < remain <= tot_gt + tot_eq
  #pragma unroll
  for (int j = 0; j < NPER; ++j) {
    if (key[j] > T) { emit(base_gt, j); ++base_gt; }
    else if (key[j] == T) { if (base_eq < eq_slots) emit(tot_gt + base_eq, j); ++base_eq; }
  }
}

// Exact select of the top `remain` of `num_ties` (key, idx) tie candidates (exact
// path): 8 keys per thread in registers. Emits POSITIONS (the caller transforms).
__device__ void hp_tie_select(const unsigned* __restrict__ tie, int num_ties, int remain,
                              int* __restrict__ out, unsigned* red) {
  const int t = threadIdx.x;
  unsigned key[8], idx[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    int p = t + i * 256;
    key[i] = (p < num_ties) ? __ldcg(tie + 2 * p) : 0u;
    idx[i] = (p < num_ties) ? __ldcg(tie + 2 * p + 1) : 0xffffffffu;
  }
  hp_bisect_select<8>(key, remain, red, [&](unsigned pos, int j) { out[pos] = (int)idx[j]; });
}

// ---------------------------------------------------------------- attention block
__device__ __forceinline__ void hp_attention_block(
    unsigned char* smem, int v, int s_i, int hg, int split, int nsa, int nblk, int nsplit,
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KV,
    const int* __restrict__ Indices, float* __restrict__ PartAcc, float* __restrict__ PartM,
    float* __restrict__ PartL, int* __restrict__ AttnCounter,
    int Skv, int H, int topk, float sm_scale, long kv_stride, int q_pos,
    int kv_fp8, long kv_rows) {
  const int t = threadIdx.x;
  const int warp = t >> 5;
  const int lane = t & 31;
  __nv_bfloat16* Qsh = (__nv_bfloat16*)smem;                    // 16 x KLD
  __nv_bfloat16* Ksh0 = Qsh + 16 * KLD;                          // 2 x BNA x KLD
  unsigned char* gbase = (unsigned char*)(Ksh0 + 2 * BNA * KLD);
  const int grp = warp >> 2;        // 0/1: tile-parity group
  const int wg = warp & 3;          // warp within the group
  const int tg = t & 127;           // thread within the group
  float* Ssh = (float*)(gbase + grp * HP_GSTATE);                 // [2][16][SLD] QK partials
  __nv_bfloat16* Psh = (__nv_bfloat16*)(Ssh + 2 * 16 * SLD);
  float* m_i = (float*)(Psh + 16 * PLD);
  float* l_i = m_i + 16;
  float* corr_sh = l_i + 16;
  int* sel_idx = (int*)(corr_sh + 16);                            // [2][BNA] ping-pong
  __nv_bfloat16* Ksh = Ksh0 + grp * BNA * KLD;
  const __nv_bfloat16* KVr = KV + (long)s_i * kv_stride;
  const int bid = 1 + grp;          // named barrier per group

  // Q -> smem: 16 rows x 72 int4, async (exact copy)
  for (int c = t; c < HB * (DQK / 8); c += 256) {
    int h = c / (DQK / 8), ch = c % (DQK / 8);
    __pipeline_memcpy_async((int4*)(Qsh + h * KLD) + ch,
                            (const int4*)(Q + (((long)s_i * H) + hg * HB + h) * DQK) + ch, sizeof(int4));
  }
  __pipeline_commit();
  if (tg < HB) { m_i[tg] = -CUDART_INF_F; l_i[tg] = 0.0f; }
  const int kps = (topk + nsa - 1) / nsa;
  const int c_lo = split * kps;
  const int c_hi = min(c_lo + kps, topk);
  const int n_tiles_all = (max(c_hi - c_lo, 0) + BNA - 1) / BNA;
  const int n_tiles = (n_tiles_all - grp + 1) / 2;   // tiles grp, grp+2, ... < n_tiles_all
  if (kv_fp8) {
    // fp8 pool path (kps == 2*BNA, launcher-checked): group g owns tile g. All
    // 256 threads gather BOTH tiles' rows into registers first (one memory
    // latency for the whole block), then dequantize into the groups' K tiles.
    // Thread (key = t&31, part = t>>5) owns nope dims [64*part, +64) (scale
    // group part>>1) and rope dims [8*part, +8) of key `key` in each tile.
    const unsigned char* pool = (const unsigned char*)KV;
    const int key = t & 31, part = t >> 5;
    uint4 nope[2][4];
    uint4 rope[2];
    float scl[2];
    #pragma unroll
    for (int tl = 0; tl < 2; ++tl) {
      int col = c_lo + tl * BNA + key;
      int slot = (col < c_hi) ? Indices[(long)s_i * topk + col] : -1;
      bool ok = (slot >= 0) && ((long)slot < kv_rows);
      const unsigned char* row = pool + (ok ? (long)slot * KV_FP8_ROW : 0L);
      const uint4* np = (const uint4*)(row + 64 * part);
      nope[tl][0] = __ldg(np);
      nope[tl][1] = __ldg(np + 1);
      nope[tl][2] = __ldg(np + 2);
      nope[tl][3] = __ldg(np + 3);
      scl[tl] = __ldg((const float*)(row + DV + 4 * (part >> 1)));
      rope[tl] = __ldg((const uint4*)(row + DV + 16 + 16 * part));
      if (part == 0) {
        // validity -> the softmax mask: 0 <= q_pos passes, Skv > q_pos drops
        int* sel_tl = (int*)((float*)(gbase + tl * HP_GSTATE) + 2 * 16 * SLD + (16 * PLD) / 2 + 3 * 16);
        sel_tl[key] = ok ? 0 : Skv;
      }
    }
    #pragma unroll
    for (int tl = 0; tl < 2; ++tl) {
      __nv_bfloat16* dst = Ksh0 + tl * BNA * KLD + key * KLD + 64 * part;
      const float s = scl[tl];
      #pragma unroll
      for (int c = 0; c < 4; ++c) {
        uint2 d0 = hp_dequant4(nope[tl][c].x, s);
        uint2 d1 = hp_dequant4(nope[tl][c].y, s);
        uint2 d2 = hp_dequant4(nope[tl][c].z, s);
        uint2 d3 = hp_dequant4(nope[tl][c].w, s);
        *(uint4*)(dst + 16 * c) = make_uint4(d0.x, d0.y, d1.x, d1.y);
        *(uint4*)(dst + 16 * c + 8) = make_uint4(d2.x, d2.y, d3.x, d3.y);
      }
      *(uint4*)(Ksh0 + tl * BNA * KLD + key * KLD + DV + 8 * part) = rope[tl];
    }
  } else if (n_tiles > 0 && tg < BNA) {
    int col = c_lo + grp * BNA + tg;
    sel_idx[tg] = (col < c_hi) ? Indices[(long)s_i * topk + col] : Skv;
  }
  __pipeline_wait_prior(0);
  __syncthreads();   // Qsh (both groups) + first sel_idx (+ fp8 K tiles) visible
  HP_STAMP(v, 1);

  const int gid = lane >> 2;
  const int tig = lane & 3;
  float po[NT_PV][4];
  #pragma unroll
  for (int nt = 0; nt < NT_PV; ++nt) po[nt][0] = po[nt][1] = po[nt][2] = po[nt][3] = 0.0f;
  const float scale2 = sm_scale * 1.44269504f;
  float* Sp = Ssh + (wg & 1) * (16 * SLD);   // this warp's partial-score plane

  for (int it = 0; it < n_tiles; ++it) {
    const int kt = grp + 2 * it;
    const int col0 = c_lo + kt * BNA;
    const int cur = it & 1;
    const int* si = sel_idx + cur * BNA;
    if (!kv_fp8) {
      for (int c = tg; c < BNA * (DQK / 8); c += 128) {
        int key = c / (DQK / 8);
        int chunk = c % (DQK / 8);
        int gidx = si[key];
        long load_idx = (gidx <= q_pos) ? (long)gidx : 0L;
        __pipeline_memcpy_async((int4*)Ksh + key * (KLD / 8) + chunk,
                                (const int4*)(KVr + load_idx * DQK) + chunk, sizeof(int4));
      }
      __pipeline_commit();
      // next tile's indices land while this tile's rows stream in
      if (it + 1 < n_tiles && tg < BNA) {
        int col = col0 + 2 * BNA + tg;
        sel_idx[(cur ^ 1) * BNA + tg] = (col < c_hi) ? Indices[(long)s_i * topk + col] : Skv;
      }
      __pipeline_wait_prior(0);
    }
    bar_sync(bid, 128);
    if (it == 0) HP_STAMP(v, 7);
    // QK, k-split: warp wg covers dims [wg*144, +144) for all 32 keys (4 n-tiles,
    // 4 independent accumulator chains of 9 mma); partials reduced across the
    // 4 warps in two deterministic levels: (p0 + p2) + (p1 + p3).
    {
      float qs[4][4];
      #pragma unroll
      for (int nt = 0; nt < 4; ++nt) qs[nt][0] = qs[nt][1] = qs[nt][2] = qs[nt][3] = 0.0f;
      #pragma unroll
      for (int i = 0; i < HP_KSPLIT; ++i) {
        int kk0 = (wg * HP_KSPLIT + i) * 16;
        unsigned a0, a1, a2, a3;
        ldmatrix_a(a0, a1, a2, a3, Qsh, KLD, 0, kk0, lane);
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          unsigned b0, b1;
          ldmatrix_b_k(b0, b1, Ksh, KLD, nt * 8, kk0, lane);
          mma_bf16_16x8x16(qs[nt][0], qs[nt][1], qs[nt][2], qs[nt][3], a0, a1, a2, a3, b0, b1);
        }
      }
      if (wg < 2) {
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          int key = nt * 8 + 2 * tig;
          Sp[gid * SLD + key] = qs[nt][0];
          Sp[gid * SLD + key + 1] = qs[nt][1];
          Sp[(gid + 8) * SLD + key] = qs[nt][2];
          Sp[(gid + 8) * SLD + key + 1] = qs[nt][3];
        }
      }
      bar_sync(bid, 128);
      if (wg >= 2) {
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
          int key = nt * 8 + 2 * tig;
          Sp[gid * SLD + key] += qs[nt][0];
          Sp[gid * SLD + key + 1] += qs[nt][1];
          Sp[(gid + 8) * SLD + key] += qs[nt][2];
          Sp[(gid + 8) * SLD + key + 1] += qs[nt][3];
        }
      }
    }
    bar_sync(bid, 128);
    if (it == 0) HP_STAMP(v, 8);
    {
      int h = wg * 4 + (lane >> 3);
      int sub = lane & 7;
      float smax = -CUDART_INF_F;
      float sv[BNA / 8];
      #pragma unroll
      for (int j = 0; j < BNA / 8; ++j) {
        int n = sub * (BNA / 8) + j;
        sv[j] = (col0 + n < topk && si[n] <= q_pos)
                    ? (Ssh[h * SLD + n] + Ssh[16 * SLD + h * SLD + n]) * scale2 : -CUDART_INF_F;
        smax = fmaxf(smax, sv[j]);
      }
      #pragma unroll
      for (int off = 1; off < 8; off <<= 1)
        smax = fmaxf(smax, __shfl_xor_sync(0xffffffffu, smax, off));
      float m_old = m_i[h];
      float m_new = fmaxf(m_old, smax);
      // fix (multi_news F1=0 at L <= topk): a tile whose keys are ALL invalid
      // (-1 incumbent padding slots when L <= topk) leaves m_old == m_new == -inf,
      // so exp2f(-inf - -inf) = exp2f(NaN) = NaN, which poisons l_i/acc (the
      // combine's w=0 cannot undo 0*NaN). Nothing accumulated yet -> corr = 1.
      float corr = (m_new > -CUDART_INF_F) ? exp2f(m_old - m_new) : 1.0f;
      float lsum = 0.0f;
      #pragma unroll
      for (int j = 0; j < BNA / 8; ++j) {
        float p = (sv[j] > -CUDART_INF_F) ? exp2f(sv[j] - m_new) : 0.0f;
        Psh[h * PLD + sub * (BNA / 8) + j] = __float2bfloat16(p);
        lsum += p;
      }
      #pragma unroll
      for (int off = 1; off < 8; off <<= 1)
        lsum += __shfl_xor_sync(0xffffffffu, lsum, off);
      if (sub == 0) {
        m_i[h] = m_new;
        corr_sh[h] = corr;
        l_i[h] = l_i[h] * corr + lsum;
      }
    }
    bar_sync(bid, 128);
    if (it == 0) HP_STAMP(v, 9);
    {
      float c_lo_f = corr_sh[gid];
      float c_hi_f = corr_sh[gid + 8];
      #pragma unroll
      for (int nt = 0; nt < NT_PV; ++nt) {
        po[nt][0] *= c_lo_f; po[nt][1] *= c_lo_f;
        po[nt][2] *= c_hi_f; po[nt][3] *= c_hi_f;
      }
      #pragma unroll
      for (int kk = 0; kk < BNA / 16; ++kk) {
        unsigned a0, a1, a2, a3;
        ldmatrix_a(a0, a1, a2, a3, Psh, PLD, 0, kk * 16, lane);
        #pragma unroll
        for (int nt = 0; nt < NT_PV; nt += 2) {
          unsigned b[2][2];
          ldmatrix_b_v_x4(b[0][0], b[0][1], b[1][0], b[1][1],
                          Ksh, KLD, kk * 16, wg * DCOL + nt * 8, lane);
          #pragma unroll
          for (int j = 0; j < 2; ++j)
            mma_bf16_16x8x16(po[nt + j][0], po[nt + j][1], po[nt + j][2], po[nt + j][3],
                             a0, a1, a2, a3, b[j][0], b[j][1]);
        }
      }
    }
    bar_sync(bid, 128);
    if (it == 0) HP_STAMP(v, 10);
    if (it == 1) HP_STAMP(v, 11);
  }
  HP_STAMP(v, 2);
  // ---- in-block merge of the two groups -> ONE partial slot per block.
  // Group 1 exports (acc, m, l) through its own (now idle) K buffer; group 0
  // rescales both onto the joint max (log2 domain, -inf-safe) and writes.
  float* xacc = (float*)(Ksh0 + BNA * KLD);   // 16 x 512 fp32 (32 KB of the 37 KB buffer)
  float* xm = xacc + HB * DV;
  float* xl = xm + HB;
  if (grp == 1) {
    #pragma unroll
    for (int nt = 0; nt < NT_PV; ++nt) {
      int col = wg * DCOL + nt * 8 + 2 * tig;
      xacc[gid * DV + col] = po[nt][0];
      xacc[gid * DV + col + 1] = po[nt][1];
      xacc[(gid + 8) * DV + col] = po[nt][2];
      xacc[(gid + 8) * DV + col + 1] = po[nt][3];
    }
    if (tg < HB) { xm[tg] = m_i[tg]; xl[tg] = l_i[tg]; }
  }
  __syncthreads();
  const long base = ((long)s_i * nblk + hg) * nsplit;
  const long slot = base + split;
  if (grp == 0) {
    float ma0 = m_i[gid], mb0 = xm[gid], mj0 = fmaxf(ma0, mb0);
    float ca0 = (ma0 > -CUDART_INF_F) ? exp2f(ma0 - mj0) : 0.0f;
    float cb0 = (mb0 > -CUDART_INF_F) ? exp2f(mb0 - mj0) : 0.0f;
    float ma1 = m_i[gid + 8], mb1 = xm[gid + 8], mj1 = fmaxf(ma1, mb1);
    float ca1 = (ma1 > -CUDART_INF_F) ? exp2f(ma1 - mj1) : 0.0f;
    float cb1 = (mb1 > -CUDART_INF_F) ? exp2f(mb1 - mj1) : 0.0f;
    #pragma unroll
    for (int nt = 0; nt < NT_PV; ++nt) {
      int col = wg * DCOL + nt * 8 + 2 * tig;
      float v0 = po[nt][0] * ca0 + xacc[gid * DV + col] * cb0;
      float v1 = po[nt][1] * ca0 + xacc[gid * DV + col + 1] * cb0;
      float v2 = po[nt][2] * ca1 + xacc[(gid + 8) * DV + col] * cb1;
      float v3 = po[nt][3] * ca1 + xacc[(gid + 8) * DV + col + 1] * cb1;
      *(float2*)(PartAcc + slot * HB * DV + (long)gid * DV + col) = make_float2(v0, v1);
      *(float2*)(PartAcc + slot * HB * DV + (long)(gid + 8) * DV + col) = make_float2(v2, v3);
    }
    if (tg < HB) {
      float ma = m_i[tg], mb = xm[tg], mj = fmaxf(ma, mb);
      float ca = (ma > -CUDART_INF_F) ? exp2f(ma - mj) : 0.0f;
      float cb = (mb > -CUDART_INF_F) ? exp2f(mb - mj) : 0.0f;
      PartM[slot * HB + tg] = mj * 0.6931471805599453f;   // natural-log domain
      PartL[slot * HB + tg] = l_i[tg] * ca + xl[tg] * cb;
    }
  }
  // publish: partial visible device-wide, then signal the row/head-group's combine blocks
  __threadfence();
  __syncthreads();
  if (t == 0) atomicAdd(&AttnCounter[(long)s_i * 256 + hg], 1);
  HP_STAMP(v, 3);
}

// ---------------------------------------------------------------- combine block
// Waits (ticket order guarantees every producer already started, so no deadlock)
// for the nsa attention blocks of (row, head-group), then reduces its 64-column
// slice of all 16 heads over the nsa partial slots with every load in flight.
__device__ __forceinline__ void hp_combine_block(
    int v, int s_i, int hg, int c, int nsa, int nblk, int nsplit,
    const float* __restrict__ PartAcc, const float* __restrict__ PartM,
    const float* __restrict__ PartL, int* __restrict__ AttnCounter,
    __nv_bfloat16* __restrict__ Out, int H) {
  __shared__ float gm[HB];
  __shared__ float gl[HB];
  __shared__ float gw[HP_MAX_NSA][HB];
  const int t = threadIdx.x;
  if (t == 0) {
    volatile int* cnt = AttnCounter + (long)s_i * 256 + hg;
    while (*cnt < nsa) __nanosleep(200);
  }
  __syncthreads();
  __threadfence();
  HP_STAMP(v, 4);
  const long base = ((long)s_i * nblk + hg) * nsplit;
  {
    const int h = t >> 4;
    const int j = t & 15;
    float mj0 = (j < nsa) ? __ldcg(PartM + (base + j) * HB + h) : -CUDART_INF_F;
    float mj1 = (j + 16 < nsa) ? __ldcg(PartM + (base + j + 16) * HB + h) : -CUDART_INF_F;
    float lj0 = (j < nsa) ? __ldcg(PartL + (base + j) * HB + h) : 0.0f;
    float lj1 = (j + 16 < nsa) ? __ldcg(PartL + (base + j + 16) * HB + h) : 0.0f;
    float m = fmaxf(mj0, mj1);
    #pragma unroll
    for (int off = 1; off < 16; off <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    float w0 = (mj0 > -CUDART_INF_F) ? __expf(mj0 - m) : 0.0f;
    float w1 = (mj1 > -CUDART_INF_F) ? __expf(mj1 - m) : 0.0f;
    float l = lj0 * w0 + lj1 * w1;
    #pragma unroll
    for (int off = 1; off < 16; off <<= 1) l += __shfl_xor_sync(0xffffffffu, l, off);
    gw[j][h] = w0;
    gw[j + 16][h] = w1;
    if (j == 0) { gm[h] = m; gl[h] = l; }
  }
  __syncthreads();
  HP_STAMP(v, 5);
  {
    const int h = t >> 4;
    const int col = c * (DV / HP_NCOMB) + (t & 15) * 4;   // 4 contiguous cols per thread
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    const float m = gm[h];
    #pragma unroll 8
    for (int j = 0; j < nsa; ++j) {
      const float w = gw[j][h];
      float4 vv = __ldcg((const float4*)(PartAcc + ((base + j) * HB + h) * DV + col));
      acc.x += (w > 0.0f) ? vv.x * w : 0.0f;
      acc.y += (w > 0.0f) ? vv.y * w : 0.0f;
      acc.z += (w > 0.0f) ? vv.z * w : 0.0f;
      acc.w += (w > 0.0f) ? vv.w * w : 0.0f;
    }
    const float l = gl[h];
    const float ld = (l > 0.0f) ? l : 1.0f;
    const bool ok = (m > -CUDART_INF_F);
    __nv_bfloat162 o0 = __floats2bfloat162_rn(ok ? acc.x / ld : 0.0f, ok ? acc.y / ld : 0.0f);
    __nv_bfloat162 o1 = __floats2bfloat162_rn(ok ? acc.z / ld : 0.0f, ok ? acc.w / ld : 0.0f);
    __nv_bfloat162* o = (__nv_bfloat162*)(Out + (((long)s_i * H) + hg * HB + h) * DV + col);
    o[0] = o0; o[1] = o1;
  }
  HP_STAMP(v, 6);
}

#define HP_QLD 136   // f16 IdxQ row stride (elements): 272 B, conflict-free 8 B fragment loads

// ---------------------------------------------------------------- indexer block
// Slice s of a row owns the INTERLEAVED set of 16-key chunks {c : c = s (mod n_idx)}
// (tile kt = chunks s + (8kt+j)*n_idx, j < 8): every contiguous window of the
// prefix is spread evenly over the row's slices (at most 16 keys per slice per
// 16*n_idx-key period), which is what makes the per-slice top-m of the
// approximate select (sel_on == 2) safe for clustered selections (recency
// windows, hot spots). Local key ordinal i -> chunk (i >> 4), key (i & 15); a
// chunk is a contiguous 2 KB segment of its page (+ 64 B of scales).
__device__ __forceinline__ void hp_indexer_block(
    unsigned char* smem, int v, int s_i, int hg, int split, int nsa, int nblk, int nsplit,
    const __nv_fp8_e4m3* __restrict__ IndexQ, const unsigned char* __restrict__ IndexK,
    const float* __restrict__ IndexW, float* __restrict__ SelSlab,
    int Hidx, const int* __restrict__ PageTable, long pt_stride, long page_bytes,
    long slab_row, int skv_row,
    int sel_on, int topk, int sel_m, unsigned* __restrict__ SelScratch, int* __restrict__ NextIdx) {
  const int t = threadIdx.x;
  const int warp = t >> 5;
  const int lane = t & 31;
  // IdxQ is held as f16 (exact e4m3 -> f16, converted ONCE per block) so the
  // B fragments feed native HMMA.16816 directly.
  unsigned short* IdxQh = (unsigned short*)smem;                        // Hidx x HP_QLD (f16 bits)
  __nv_fp8_e4m3* IdxKsh = (__nv_fp8_e4m3*)(IdxQh + Hidx * HP_QLD);      // HP_ISTAGES x HP_BNI x IKLD
  float* sel_w = (float*)(IdxKsh + HP_ISTAGES * HP_BNI * IKLD);         // Hidx
  float* IdxScl = sel_w + Hidx;                                        // HP_ISTAGES x HP_BNI page-resident key scales
  int* pt_s = (int*)(IdxScl + HP_ISTAGES * HP_BNI);                    // 2 * HP_MAX_TILES page ids
  unsigned* hist_s = (unsigned*)(pt_s + 2 * HP_MAX_TILES);             // HP_HSIZE coarse histogram
  unsigned* misc_s = hist_s + HP_HSIZE;                                // 32 words (select bookkeeping)
  float* lg_s = (float*)(misc_s + 32);                                 // HP_LG_KEYS slice logits (select)
  const int n_idx = nblk * (nsplit - nsa);
  const int slice = hg * (nsplit - nsa) + (split - nsa);
  const int prefix = skv_row;
  const int* PT = PageTable + (long)s_i * pt_stride;
  float* slab = SelSlab + (long)s_i * slab_row;
  // slab tail [prefix, slab_row) -> -inf, spread over the row's indexer blocks
  {
    long tail = slab_row - prefix;
    if (tail > 0) {
      long tps = (tail + n_idx - 1) / n_idx;
      long p0 = prefix + (long)slice * tps;
      long p1 = min(p0 + tps, slab_row);
      for (long p = p0 + t; p < p1; p += 256) slab[p] = -CUDART_INF_F;
    }
  }
  const int nchunks = (prefix + 15) >> 4;
  const int my_chunks = (slice < nchunks) ? (nchunks - slice + n_idx - 1) / n_idx : 0;
  const int n_tiles = (my_chunks + 7) >> 3;
  const int n_loc = my_chunks * 16;                // local ordinals (some beyond the prefix)
  const bool lg_in_smem = (n_loc <= HP_LG_KEYS);   // else the select re-reads the slab from L2
  const int diag = hp_diag_mode;
  const bool wg = !(diag & 4);   // wgmma fp8 scan (default); bit2 of hp_diag_mode = legacy f16 HMMA path
  // wgmma operands: 1024-B aligned region at the front of smem (Q 8 KB, then HP_ISTAGES stages)
  const unsigned smem_u = (unsigned)__cvta_generic_to_shared(smem);
  const unsigned wg_base = (smem_u + 1023u) & ~1023u;
  unsigned char* wgQ = smem + (wg_base - smem_u);
  unsigned char* wgK = wgQ + HP_WG_QBYTES;
  // page ids of this slice -> smem (one coalesced read instead of a dependent
  // global load in front of every prefetch); histogram cleared
  for (int i = t; i < 8 * n_tiles; i += 256) {
    int ch = slice + i * n_idx;                    // global 16-key chunk -> its page
    pt_s[i] = (i < my_chunks) ? PT[ch >> 2] : 0;
  }
  for (int i = t; i < HP_HSIZE; i += 256) hist_s[i] = 0u;
  __syncthreads();
  // Async copy of tile kt (its <=2 allocated pages + their 64 fp32 key scales)
  // into stage `buf`. Bytes of keys beyond the prefix are never consumed (each
  // key's logit is its own mma row), so whole pages are copied.
#define HP_PREFETCH(kt, buf) do {                                                    \
    unsigned char* _dst = (unsigned char*)(IdxKsh + (buf) * (HP_BNI * IKLD));         \
    unsigned char* _wdst = wgK + (buf) * HP_WG_STAGE;                                 \
    _Pragma("unroll")                                                                \
    for (int _i = 0; _i < 4; ++_i) {                                                 \
      const int _q = t + 256 * _i;             /* 16-B chunk id of the tile: row = q>>3 */ \
      const int _row = _q >> 3, _chk = _q & 7;                                       \
      const int _j = 8 * (kt) + (_row >> 4);   /* segment (16-key chunk) of this row */ \
      if (_j < my_chunks) {                                                          \
        const int _ch = slice + _j * n_idx;                                          \
        const unsigned char* _src = IndexK + (long)pt_s[_j] * page_bytes            \
                                  + (long)(_ch & 3) * 16 * DIDX + (_row & 15) * DIDX + (_chk << 4); \
        void* _d = wg ? (void*)(_wdst + _row * 128 + ((_chk ^ (_row & 7)) << 4))     \
                      : (void*)(_dst + _row * IKLD + (_chk << 4));                   \
        __pipeline_memcpy_async(_d, _src, sizeof(int4));                             \
      }                                                                              \
    }                                                                                \
    if (t < 32) {                              /* 8 segments x 64 B of key scales */  \
      const int _j = 8 * (kt) + (t >> 2);                                            \
      if (_j < my_chunks) {                                                          \
        const int _ch = slice + _j * n_idx;                                          \
        __pipeline_memcpy_async((int4*)(IdxScl + (buf) * HP_BNI + (t >> 2) * 16) + (t & 3), \
                                (const int4*)(IndexK + (long)pt_s[_j] * page_bytes + 64 * DIDX \
                                              + (long)(_ch & 3) * 16 * 4) + (t & 3), sizeof(int4)); \
      }                                                                              \
    }                                                                                \
    __pipeline_commit();                                                             \
  } while (0)
  // prefetches FIRST (they do not depend on Q), then the Q conversion overlaps them
  for (int p = 0; p < HP_ISTAGES - 1; ++p)
    if (p < n_tiles && !(diag & 2)) HP_PREFETCH(p, p);
  {
    // all of this thread's IdxQ words are loaded up front (Hidx <= 64 -> <= 8 per
    // thread), so the block pays ONE global latency for the query, not eight
    const int nq = Hidx * (DIDX / 4);
    const unsigned* qsrc = (const unsigned*)(IndexQ + (long)s_i * Hidx * DIDX);
    unsigned wq[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) { int c = t + i * 256; wq[i] = (c < nq) ? __ldg(qsrc + c) : 0u; }
    float wsel = (t < Hidx) ? IndexW[(long)s_i * Hidx + t] : 0.0f;
    // IdxQ -> f16 smem, PERMUTED per 16-dim group so that one ldmatrix.x4 yields
    // the B fragments in the fp8-word slotting hp_index_mma_k32 expects:
    // 8x8 matrix 0 = pairs (4j, 4j+1) [lo halves], matrix 1 = pairs (4j+2, 4j+3)
    // [hi halves] of dims 16m..16m+15; word ch (dims 4ch..4ch+3) -> lo at
    // position 16m + 2j, hi at 16m + 8 + 2j (m = ch/4, j = ch%4).
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      int c = t + i * 256;
      if (c < nq) {
        int h = c / (DIDX / 4), ch = c % (DIDX / 4);
        if (wg) {
          // raw e4m3 word (dims 4ch..4ch+3) into the 128B-swizzled K-major Q tile
          int chunk = ch >> 2;
          *(unsigned*)(wgQ + h * 128 + ((chunk ^ (h & 7)) << 4) + ((ch & 3) << 2)) = wq[i];
        } else {
          uint2 f = hp_e4m3x4_to_f16x4(wq[i]);
          unsigned* row = (unsigned*)(IdxQh + h * HP_QLD);
          int m = ch >> 2, j = ch & 3;
          row[(16 * m + 2 * j) >> 1] = f.x;
          row[(16 * m + 8 + 2 * j) >> 1] = f.y;
        }
      }
    }
    if (t < Hidx) sel_w[t] = wsel;
  }
  HP_STAMP(v, 1);
  const int gid = lane >> 2;
  const int tid = lane & 3;
  const int mkey0 = warp * 16;
  const int n_ntiles = Hidx / 8;
  for (int kt = 0; kt < n_tiles; ++kt) {
    const int buf = kt % HP_ISTAGES;
    // groups committed after tile kt's: min(HP_ISTAGES-2, n_tiles-1-kt) may stay in flight
    int pending = min(HP_ISTAGES - 2, n_tiles - 1 - kt);
    if (pending >= 2) __pipeline_wait_prior(2);
    else if (pending == 1) __pipeline_wait_prior(1);
    else __pipeline_wait_prior(0);
    __syncthreads();   // tile kt visible; everyone done with tile kt-1 (its stage is free); Q visible
    if (kt == 1) HP_STAMP(v, 7);
    if (kt + HP_ISTAGES - 1 < n_tiles && !(diag & 2)) HP_PREFETCH(kt + HP_ISTAGES - 1, (kt + HP_ISTAGES - 1) % HP_ISTAGES);
    const __nv_fp8_e4m3* Kb = IdxKsh + buf * (HP_BNI * IKLD);
    // key ordinal n of this tile -> local ordinal i = 128*kt + n, chunk = i >> 4
    const int chunk0 = slice + 8 * kt * n_idx;   // chunk of n < 16; segment j -> chunk0 + j*n_idx
    if (diag & 1) {   // memory floor (gate diagnostic): no fragments, no mma
      if (tid == 0) {
        float acc = (float)((const unsigned char*)Kb)[(mkey0 + gid) * IKLD + lane];
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
          int n = mkey0 + gid + half * 8;
          int g = ((chunk0 + (n >> 4) * n_idx) << 4) + (n & 15);
          if (g < prefix) slab[g] = acc * IdxScl[buf * HP_BNI + n];
        }
      }
      continue;
    }
    float accA = 0.0f, accB = 0.0f;   // keys mkey0+gid, mkey0+gid+8
    if (wg) {
      // one warpgroup per 64-key half: D[64 x Hidx] = K[64 x 128] . Q[Hidx x 128]^T in 4 k32 steps
      float d[32];
      #pragma unroll
      for (int i = 0; i < 32; ++i) d[i] = 0.0f;
      const unsigned a_base = wg_base + HP_WG_QBYTES + buf * HP_WG_STAGE + (warp >> 2) * (64 * 128);
      const unsigned b_base = wg_base;
      hp_wg_fence();
      #pragma unroll
      for (int kk = 0; kk < DIDX / 32; ++kk)
        hp_wgmma_64x64x32(d, hp_wg_desc(a_base + kk * 32), hp_wg_desc(b_base + kk * 32), 1);
      hp_wg_commit();
      hp_wg_wait0();
      // same fragment layout as mma.sync: d[4j..4j+3] = rows (gid, gid+8), cols 8j+2tid..+1
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (j * 8 < Hidx)
          hp_index_reduce_ntile(accA, accB, d[4 * j], d[4 * j + 1], d[4 * j + 2], d[4 * j + 3],
                                sel_w[j * 8 + 2 * tid], sel_w[j * 8 + 2 * tid + 1]);
      }
    } else {
    // A fragments: the 4 fp8 words per k32 step, each unpacked to (lo, hi) f16x2
    // exactly as ptxas's e4m3 mma emulation slots them: lo -> k-slots (2tid, 2tid+1),
    // hi -> (8+2tid, 8+2tid+1) of one native m16n8k16 (see hp_index_mma_k32).
    uint2 a[DIDX / 32][4];
    #pragma unroll
    for (int kk = 0; kk < DIDX / 32; ++kk) {
      const __nv_fp8_e4m3* Ab = Kb + (mkey0 + gid) * IKLD + kk * 32 + tid * 4;
      a[kk][0] = hp_e4m3x4_to_f16x4(*(const unsigned int*)(Ab));
      a[kk][1] = hp_e4m3x4_to_f16x4(*(const unsigned int*)(Ab + 8 * IKLD));
      a[kk][2] = hp_e4m3x4_to_f16x4(*(const unsigned int*)(Ab + 16));
      a[kk][3] = hp_e4m3x4_to_f16x4(*(const unsigned int*)(Ab + 8 * IKLD + 16));
    }
    #pragma unroll 8
    for (int ntile = 0; ntile < n_ntiles; ++ntile) {
      int n0 = ntile * 8;
      float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
      #pragma unroll
      for (int kk = 0; kk < DIDX / 32; ++kk) {
        // x4: matrices 0,1 = dims 32kk..+15 (lo/hi pairs), 2,3 = dims 32kk+16..+31
        const unsigned short* pB = IdxQh + (n0 + (lane & 7)) * HP_QLD + kk * 32 + (lane >> 3) * 8;
        unsigned addr = (unsigned)__cvta_generic_to_shared(pB);
        uint2 b0, b1;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(b0.x), "=r"(b0.y), "=r"(b1.x), "=r"(b1.y) : "r"(addr));
        hp_index_mma_k32(c0, c1, c2, c3, a[kk][0], a[kk][1], a[kk][2], a[kk][3], b0, b1);
      }
      hp_index_reduce_ntile(accA, accB, c0, c1, c2, c3, sel_w[n0 + 2 * tid], sel_w[n0 + 2 * tid + 1]);
    }
    }   // legacy path
    if (kt == 1) HP_STAMP(v, 8);
    accA = hp_index_reduce_lanes(accA);
    accB = hp_index_reduce_lanes(accB);
    if (tid == 0) {
      #pragma unroll
      for (int half = 0; half < 2; ++half) {
        int n = mkey0 + gid + half * 8;
        int g = ((chunk0 + (n >> 4) * n_idx) << 4) + (n & 15);
        if (g < prefix) {
          float acc = half ? accB : accA;
          acc *= IdxScl[buf * HP_BNI + n];   // page-resident per-key fp8 scale (prefetched)
          slab[g] = acc;
          if (sel_on) {
            if (lg_in_smem) lg_s[kt * HP_BNI + n] = acc;
            atomicAdd(&hist_s[hp_coarse_bin(acc)], 1u);
          }
        }
      }
    }
    if (kt == 0) HP_STAMP(v, 2);
    if (kt == 1) HP_STAMP(v, 9);
  }
  HP_STAMP(v, 3);
#undef HP_PREFETCH
  if (!sel_on) return;
  // local ordinal i -> prefix position (valid iff (i >> 4) < my_chunks && g < prefix)
#define HP_G_OF(i) (((slice + ((i) >> 4) * n_idx) << 4) + ((i) & 15))
#define HP_VAL_OF(i) (lg_in_smem ? lg_s[(i)] : __ldcg(slab + HP_G_OF(i)))

  unsigned* G = SelScratch + (long)s_i * HP_SEL_STRIDE;
  int* out = NextIdx + (long)s_i * topk;
  __syncthreads();   // hist_s / lg_s complete

  if (sel_on == 2) {
    // ================= approximate: per-slice top-m -> last-arriver exact merge ====
    if (prefix <= topk) {
      // trivial row: every scored position is selected (no candidates needed)
      for (int i = t; i < n_loc; i += 256) { int g = HP_G_OF(i); if (g < prefix) out[g] = hp_page_to_slot(PT, g); }
    } else {
      // local threshold bin for m over this slice's coarse histogram
      unsigned h4[4], lsum = 0;
      #pragma unroll
      for (int q = 0; q < 4; ++q) { h4[q] = hist_s[4 * t + q]; lsum += h4[q]; }
      unsigned inc = lsum;
      #pragma unroll
      for (int off = 1; off < 32; off <<= 1) { unsigned n = __shfl_up_sync(0xffffffffu, inc, off); if (lane >= off) inc += n; }
      if (lane == 31) misc_s[8 + warp] = inc;
      if (t == 0) { misc_s[0] = 0xffffffffu; misc_s[1] = 0; misc_s[2] = 0; misc_s[3] = 0; }
      __syncthreads();
      unsigned before = 0, total = 0;
      #pragma unroll
      for (int w = 0; w < 8; ++w) { before += (w < warp) ? misc_s[8 + w] : 0u; total += misc_s[8 + w]; }
      {
        unsigned pre = before + inc - lsum;
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
          pre += h4[q];
          unsigned above = total - pre;
          if (above < (unsigned)sel_m && above + h4[q] >= (unsigned)sel_m) { misc_s[0] = (unsigned)(4 * t + q); misc_s[3] = above; }
        }
      }
      __syncthreads();
      const unsigned thr = misc_s[0];
      const unsigned above0 = misc_s[3];
      unsigned* cand = G + HP_SEL_CAND + (long)slice * 2 * sel_m;
      if (total <= (unsigned)sel_m) {
        // fewer scored keys than m: emit all, pad the rest with (key 0, idx -1)
        for (int i0 = 0; i0 < n_loc; i0 += 256) {
          int i = i0 + t;
          int g = (i < n_loc) ? HP_G_OF(i) : prefix;
          bool ok = g < prefix;
          unsigned p = hp_warp_alloc(&misc_s[1], ok);
          if (ok) { cand[2 * p] = hp_exact_key(HP_VAL_OF(i)); cand[2 * p + 1] = (unsigned)g; }
        }
        __syncthreads();
        for (unsigned p = misc_s[1] + t; p < (unsigned)sel_m; p += 256) { cand[2 * p] = 0u; cand[2 * p + 1] = 0xffffffffu; }
      } else {
        // above-bin keys always; threshold-bin keys fill the remaining m - above slots
        for (int i0 = 0; i0 < n_loc; i0 += 256) {
          int i = i0 + t;
          int g = (i < n_loc) ? HP_G_OF(i) : prefix;
          bool ok = g < prefix;
          float val = ok ? HP_VAL_OF(i) : 0.0f;
          unsigned b = ok ? hp_coarse_bin(val) : 0u;
          bool up = ok && b > thr, eq = ok && b == thr;
          unsigned pa = hp_warp_alloc(&misc_s[1], up);
          unsigned pe = above0 + hp_warp_alloc(&misc_s[2], eq);
          if (up) { cand[2 * pa] = hp_exact_key(val); cand[2 * pa + 1] = (unsigned)g; }
          else if (eq && pe < (unsigned)sel_m) { cand[2 * pe] = hp_exact_key(val); cand[2 * pe + 1] = (unsigned)g; }
        }
      }
    }
    HP_STAMP(v, 4);    // local top-m emitted
    __threadfence();
    __syncthreads();
    if (t == 0) misc_s[5] = (atomicAdd(&G[HP_SEL_DONE], 1u) == (unsigned)(n_idx - 1)) ? 1u : 0u;
    __syncthreads();
    if (misc_s[5] == 0) return;
    __threadfence();
    HP_STAMP(v, 5);
    if (prefix <= topk) {
      for (int p = prefix + t; p < topk; p += 256) out[p] = -1;
    } else {
      // exact top-k merge over the n_idx*m candidates: keys register-resident
      // (candidate c = t + 256*j), bisection select, no atomics
      const int ncand = n_idx * sel_m;
      const unsigned* src = G + HP_SEL_CAND;
      unsigned kreg[32];
      #pragma unroll
      for (int j = 0; j < 32; ++j) { int c = t + 256 * j; kreg[j] = (c < ncand) ? __ldcg(src + 2 * c) : 0u; }
      HP_STAMP(v, 12);   // candidates staged
      hp_bisect_select<32>(kreg, topk, hist_s, [&](unsigned pos, int j) { out[pos] = t + 256 * j; });
      __syncthreads();
      HP_STAMP(v, 13);   // select done
      // transform pass: candidate id -> position (candidate buffer) -> KV slot, with
      // independent gathers (a gather inside the emit loop serializes on latency)
      {
        int ids[8], pos[8], pg8[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) { int p = t + i * 256; ids[i] = (p < topk) ? out[p] : -1; }
        #pragma unroll
        for (int i = 0; i < 8; ++i) pos[i] = (ids[i] >= 0) ? (int)__ldcg(src + 2 * ids[i] + 1) : -1;
        #pragma unroll
        for (int i = 0; i < 8; ++i) pg8[i] = (pos[i] >= 0) ? PT[pos[i] >> 6] : -1;
        #pragma unroll
        for (int i = 0; i < 8; ++i) { int p = t + i * 256; if (p < topk) out[p] = (pos[i] >= 0) ? ((pg8[i] << 6) | (pos[i] & 63)) : -1; }
      }
      HP_STAMP(v, 14);   // transformed
    }
    __syncthreads();
    if (t == 0) G[HP_SEL_DONE] = 0u;
    __threadfence();
    HP_STAMP(v, 6);
    return;
  }

  // ================= exact: row-cooperative selection ========================
  // Phase 1: publish this slice's coarse histogram; arrive; wait for the row.
  for (int b = t; b < HP_HSIZE; b += 256) {
    unsigned c = hist_s[b];
    if (c) atomicAdd(&G[b], c);
  }
  __threadfence();
  __syncthreads();
  if (t == 0) {
    atomicAdd(&G[HP_SEL_ARRIVE], 1u);
    while (hp_ld_acquire(&G[HP_SEL_ARRIVE]) < (unsigned)n_idx) __nanosleep(128);
  }
  __syncthreads();
  __threadfence();
  HP_STAMP(v, 4);
  // Phase 2: threshold bin of the ROW (all blocks compute the same value).
  unsigned h4[4];
  unsigned lsum = 0;
  #pragma unroll
  for (int i = 0; i < 4; ++i) { h4[i] = __ldcg(&G[4 * t + i]); lsum += h4[i]; }
  unsigned inc = lsum;
  #pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    unsigned n = __shfl_up_sync(0xffffffffu, inc, off);
    if (lane >= off) inc += n;
  }
  if (lane == 31) misc_s[8 + warp] = inc;
  if (t == 0) { misc_s[0] = 0xffffffffu; misc_s[1] = 0; misc_s[2] = 0; misc_s[3] = 0; misc_s[4] = 0; }
  __syncthreads();
  unsigned before = 0, total = 0;
  #pragma unroll
  for (int w = 0; w < 8; ++w) { before += (w < warp) ? misc_s[8 + w] : 0u; total += misc_s[8 + w]; }
  {
    unsigned pre = before + inc - lsum;   // exclusive prefix through bin 4t-1
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      pre += h4[i];
      unsigned above = total - pre;
      if (above < (unsigned)topk && above + h4[i] >= (unsigned)topk) misc_s[0] = (unsigned)(4 * t + i);
    }
  }
  __syncthreads();
  const unsigned thr = misc_s[0];
  unsigned* s_cnt_gt = misc_s + 1;   // this block's above count
  unsigned* s_cnt_eq = misc_s + 2;   // this block's tie count
  if (total <= (unsigned)topk) {
    // trivial row (prefix <= topk): every scored position is selected
    for (int i = t; i < n_loc; i += 256) { int g = HP_G_OF(i); if (g < prefix) out[g] = hp_page_to_slot(PT, g); }
  } else {
    // pass A: count this block's above / tie elements (classified by the SAME bin
    // function that built the histogram, so the counts are exactly consistent)
    unsigned n_gt = 0, n_eq = 0;
    for (int i = t; i < n_loc; i += 256) {
      if (HP_G_OF(i) >= prefix) continue;
      unsigned b = hp_coarse_bin(HP_VAL_OF(i));
      n_gt += (b > thr); n_eq += (b == thr);
    }
    unsigned wg = n_gt, we = n_eq;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      wg += __shfl_up_sync(0xffffffffu, wg, off) * (lane >= off);
      we += __shfl_up_sync(0xffffffffu, we, off) * (lane >= off);
    }
    if (lane == 31) { misc_s[16 + warp] = wg; misc_s[24 + warp] = we; }
    __syncthreads();
    unsigned bg = 0, be = 0, tg_ = 0, te_ = 0;
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
      bg += (w < warp) ? misc_s[16 + w] : 0u; tg_ += misc_s[16 + w];
      be += (w < warp) ? misc_s[24 + w] : 0u; te_ += misc_s[24 + w];
    }
    if (t == 0) {
      *s_cnt_gt = atomicAdd(&G[HP_SEL_CNT_GT], tg_);   // row base of this block's above range
      *s_cnt_eq = atomicAdd(&G[HP_SEL_CNT_EQ], te_);   // row base of this block's tie range
    }
    __syncthreads();
    // pass B: emit above elements (slot space) and stash tie candidates
    unsigned pg = *s_cnt_gt + bg + wg - n_gt;   // this thread's first above position
    unsigned pe = *s_cnt_eq + be + we - n_eq;   // this thread's first tie position
    unsigned* tie = G + HP_SEL_TIE;
    for (int i = t; i < n_loc; i += 256) {
      int g = HP_G_OF(i);
      if (g >= prefix) continue;
      float val = HP_VAL_OF(i);
      unsigned b = hp_coarse_bin(val);
      if (b > thr) {
        if (pg < (unsigned)topk) out[pg] = g;        // position; the last block transforms
        ++pg;
      } else if (b == thr) {
        if (pe < HP_MAXTIE) { tie[2 * pe] = hp_exact_key(val); tie[2 * pe + 1] = (unsigned)g; }
        ++pe;
      }
    }
  }
  // Phase 3: last arriver resolves the ties (or pads), then re-arms the row.
  __threadfence();
  __syncthreads();
  if (t == 0) misc_s[5] = (atomicAdd(&G[HP_SEL_DONE], 1u) == (unsigned)(n_idx - 1)) ? 1u : 0u;
  __syncthreads();
  if (misc_s[5] == 0) return;
  __threadfence();
  HP_STAMP(v, 5);
  if (total <= (unsigned)topk) {
    for (int p = (int)total + t; p < topk; p += 256) out[p] = -1;
  } else {
    const unsigned gt = min(hp_ld_acquire(&G[HP_SEL_CNT_GT]), (unsigned)topk);
    const unsigned eq = min(hp_ld_acquire(&G[HP_SEL_CNT_EQ]), (unsigned)HP_MAXTIE);
    const unsigned remain = (gt < (unsigned)topk) ? (unsigned)topk - gt : 0u;
    if (remain > 0) {
      if (eq >= remain) {
        // warp-private histograms live in the (now dead) slice-logit region
        hp_tie_select(G + HP_SEL_TIE, (int)eq, (int)remain, out + gt, hist_s);
      } else {
        // more slots than candidates (ties overflowed HP_MAXTIE): emit all, pad -1
        const unsigned* tie = G + HP_SEL_TIE;
        for (unsigned p = t; p < eq; p += 256) out[gt + p] = (int)__ldcg(tie + 2 * p + 1);
        for (unsigned p = eq + t; p < remain; p += 256) out[gt + p] = -1;
      }
    }
    __syncthreads();
    __threadfence();
    // transform pass over the whole row: position -> KV slot (entries written by the
    // other blocks are read through L2)
    {
      int pos[8], pg8[8];
      #pragma unroll
      for (int i = 0; i < 8; ++i) { int p = t + i * 256; pos[i] = (p < topk) ? __ldcg(out + p) : -1; }
      #pragma unroll
      for (int i = 0; i < 8; ++i) pg8[i] = (pos[i] >= 0) ? PT[pos[i] >> 6] : -1;
      #pragma unroll
      for (int i = 0; i < 8; ++i) { int p = t + i * 256; if (p < topk) out[p] = (pos[i] >= 0) ? ((pg8[i] << 6) | (pos[i] & 63)) : -1; }
    }
  }
  __syncthreads();
  for (int b = t; b < HP_HSIZE; b += 256) G[b] = 0u;
  if (t == 0) { G[HP_SEL_ARRIVE] = 0u; G[HP_SEL_CNT_GT] = 0u; G[HP_SEL_CNT_EQ] = 0u; G[HP_SEL_DONE] = 0u; }
  __threadfence();
  HP_STAMP(v, 6);
#undef HP_G_OF
#undef HP_VAL_OF
}

// ---------------------------------------------------------------- the kernel
// grid = [S*nblk*nsa attention][S*nblk*(nsplit-nsa) indexer][S*nblk*HP_NCOMB combine]
// blocks, in blockIdx order. Attention (latency-critical) is dispatched first;
// the combine blocks are LAST, so every producer they wait for has a lower
// blockIdx and was dispatched earlier (in-order CTA dispatch, the same
// assumption CUTLASS stream-K's separate-reduction fixup relies on), and the
// spin-wait cannot deadlock.
extern "C" __global__ void __launch_bounds__(256, 2) hide_paged_kernel(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ KV,
    const int* __restrict__ Indices, const __nv_fp8_e4m3* __restrict__ IndexQ,
    const unsigned char* __restrict__ IndexK,   // paged index-K buffer
    const float* __restrict__ IndexW,
    float* __restrict__ PartAcc, float* __restrict__ PartM, float* __restrict__ PartL,
    float* __restrict__ SelSlab,
    int* __restrict__ AttnCounter,   // [S, 256] zero at first launch; self-resetting
    __nv_bfloat16* __restrict__ Out,
    int S, int Skv, int H, int Hidx, int topk, int nsplit, float sm_scale,
    int with_select, int attn_on, long kv_stride,
    const int* __restrict__ SeqLens, const int* __restrict__ PageTable,
    long pt_stride, long page_bytes, long slab_stride, int nsplit_attn,
    int kv_fp8, long kv_rows, int sel_on, unsigned* __restrict__ SelScratch,
    int* __restrict__ NextIdx, int sel_m) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int v = blockIdx.x;
  HP_STAMP(v, 0);
  const int nblk = H / HB;
  const int nsa = attn_on ? nsplit_attn : 0;
  const int n_attn = S * nblk * nsa;
  const int n_idx_all = S * nblk * (nsplit - nsa);
  if (v < n_attn) {
    const int split = v % nsa;
    const int tmp = v / nsa;
    const int hg = tmp % nblk;
    const int s_i = tmp / nblk;
    const int skv_row = SeqLens[s_i];
    hp_attention_block(smem, v, s_i, hg, split, nsa, nblk, nsplit, Q, KV, Indices, PartAcc,
                       PartM, PartL, AttnCounter, Skv, H, topk, sm_scale, kv_stride, skv_row - 1,
                       kv_fp8, kv_rows);
  } else if (v < n_attn + n_idx_all) {
    const int u = v - n_attn;
    const int split = nsa + u % (nsplit - nsa);
    const int tmp = u / (nsplit - nsa);
    const int hg = tmp % nblk;
    const int s_i = tmp / nblk;
    const int skv_row = SeqLens[s_i];
    if (with_select == 1)
      hp_indexer_block(smem, v, s_i, hg, split, nsa, nblk, nsplit, IndexQ, IndexK, IndexW, SelSlab,
                       Hidx, PageTable, pt_stride, page_bytes, slab_stride, skv_row,
                       sel_on, topk, sel_m, SelScratch, NextIdx);
  } else {
    const int cidx = v - n_attn - n_idx_all;
    const int c = cidx % HP_NCOMB;
    const int rh = cidx / HP_NCOMB;
    const int s_i = rh / nblk, hg = rh % nblk;
    hp_combine_block(v, s_i, hg, c, nsa, nblk, nsplit, PartAcc, PartM, PartL, AttnCounter, Out, H);
    // the last combine block of (row, head-group) re-arms both counters
    __syncthreads();
    if (threadIdx.x == 0) {
      int prev = atomicAdd(&AttnCounter[(long)s_i * 256 + 16 + hg], 1);
      if (prev == HP_NCOMB - 1) {
        AttnCounter[(long)s_i * 256 + hg] = 0;
        AttnCounter[(long)s_i * 256 + 16 + hg] = 0;
        __threadfence();
      }
    }
  }
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
void launch_fused_sparse_splitk(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexK, torch::Tensor IndexW, torch::Tensor IndexKScale,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlab,
    torch::Tensor SelHist,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int nsplit,
    double sm_scale, int with_select, int attn_on, int win);
void launch_fused_sparse_splitk_selectfold(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexK, torch::Tensor IndexW, torch::Tensor IndexKScale,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlab,
    torch::Tensor SelHist, torch::Tensor NextIdx, torch::Tensor SelCounter,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int nsplit,
    double sm_scale, int win);
void launch_fused_sparse_splitk_tapw(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexK, torch::Tensor IndexW, torch::Tensor IndexKScale,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlab,
    torch::Tensor SelHist, torch::Tensor TapW,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int nsplit,
    double sm_scale, int win);
void launch_fused_sparse_splitk_pool(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexK, torch::Tensor IndexW, torch::Tensor IndexKScale,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlab,
    torch::Tensor SelHist, torch::Tensor TapW,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int attend_k,
    int nsplit, double sm_scale, int win);
void launch_fused_sparse_splitk_batch(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexK, torch::Tensor IndexW, torch::Tensor IndexKScale,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlab,
    torch::Tensor SelHist, torch::Tensor TapW,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int attend_k,
    int nsplit, double sm_scale, int with_select, int win, int64_t kv_stride);
void launch_fused_sparse_splitk_batch_idx(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexK, torch::Tensor IndexW, torch::Tensor IndexKScale,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlab,
    torch::Tensor SelHist,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int nsplit,
    double sm_scale, int with_select, int attn_on, int win, int64_t kv_stride,
    int64_t idx_k_stride, int64_t idx_scale_stride);
void launch_fused_sparse_splitk_paged(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexKPaged, torch::Tensor IndexW,
    torch::Tensor SeqLens, torch::Tensor PageTable,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlab,
    torch::Tensor SelHist,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int nsplit,
    double sm_scale, int with_select, int attn_on, int64_t kv_stride,
    int64_t pt_stride, int64_t page_bytes, int64_t slab_stride, int nsplit_attn,
    int kv_fp8, int64_t kv_rows, int sel_on, torch::Tensor SelScratch, torch::Tensor NextIdx, int sel_m);
void launch_cascade_epilogue(torch::Tensor Planes, torch::Tensor Indices,
    torch::Tensor TapOut, int S, int nblk, int topk, int win, int Skv);
void launch_cascade_epilogue_batch(torch::Tensor Planes, torch::Tensor Indices,
    torch::Tensor TapOut, int S, int nblk, int topk, int win, int Skv);
void launch_cascade_select(torch::Tensor Planes, torch::Tensor Indices,
    torch::Tensor NextIdx, int S, int nblk, int topk, int win, int Skv);
void hide_debug_enable(int on);
torch::Tensor hide_debug_read();
void hide_diag_mode(int mode);
void launch_fused_sparse_splitk_exptap(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexK, torch::Tensor IndexW, torch::Tensor IndexKScale,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlabH,
    torch::Tensor SelHist,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int nsplit,
    double sm_scale, int win);
void launch_fused_sparse_splitk_exptap_batch(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexK, torch::Tensor IndexW, torch::Tensor IndexKScale,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlabH,
    torch::Tensor SelHist,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int nsplit,
    double sm_scale, int win, int64_t kv_stride);
void launch_cascade_exp_epilogue(torch::Tensor PlanesH, torch::Tensor PartM, torch::Tensor PartL,
    torch::Tensor Indices, torch::Tensor TapOut, torch::Tensor ExpStats, int S, int nblk,
    int topk, int win, int Skv, int nsplit);
void launch_cascade_exp_epilogue_batch(torch::Tensor PlanesH, torch::Tensor PartM, torch::Tensor PartL,
    torch::Tensor Indices, torch::Tensor TapOut, torch::Tensor ExpStats, int S, int nblk,
    int topk, int win, int Skv, int nsplit);
"""

LAUNCH_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
extern "C" __global__ void fused_sparse_splitk_kernel(const __nv_bfloat16*, const __nv_bfloat16*,
    const int*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, const float*, const float*, const float*,
    float*, float*, float*, float*, int*, int*, int*, int, int, int, int, int, int, int, float, int, int, int, long,
    long, long, const int*, const int*, long, long, long, int);
extern "C" __global__ void combine_kernel(const float*, const float*, const float*,
    __nv_bfloat16*, int, int, int, int);
extern "C" __global__ void hide_paged_kernel(const __nv_bfloat16*, const __nv_bfloat16*,
    const int*, const __nv_fp8_e4m3*, const unsigned char*, const float*,
    float*, float*, float*, float*, int*, __nv_bfloat16*,
    int, int, int, int, int, int, float, int, int, long,
    const int*, const int*, long, long, long, int,
    int, long, int, unsigned*, int*, int);
extern "C" __global__ void cascade_epilogue_kernel(const float*, const int*, float*,
    int, int, int, int, int, int);
extern "C" __global__ void cascade_exp_epilogue_kernel(const float*, const float*, const float*,
    const int*, float*, float*, int, int, int, int, int, int, int);
extern "C" __global__ void cascade_select_kernel(const float*, const int*, int*,
    int, int, int, int, int);
void hide_debug_enable(int on) {
  cudaError_t e = cudaMemcpyToSymbol(hp_dbg_on, &on, sizeof(int));
  TORCH_CHECK(e == cudaSuccess, "hide_debug_enable failed: ", cudaGetErrorString(e));
  if (on) {
    unsigned long long* p = nullptr;
    e = cudaGetSymbolAddress((void**)&p, hp_dbg);
    TORCH_CHECK(e == cudaSuccess, "hp_dbg symbol lookup failed: ", cudaGetErrorString(e));
    e = cudaMemset(p, 0, sizeof(unsigned long long) * 512 * 16);
    TORCH_CHECK(e == cudaSuccess, "hp_dbg memset failed: ", cudaGetErrorString(e));
  }
}
void hide_diag_mode(int mode) {
  cudaError_t e = cudaMemcpyToSymbol(hp_diag_mode, &mode, sizeof(int));
  TORCH_CHECK(e == cudaSuccess, "hide_diag_mode failed: ", cudaGetErrorString(e));
}
torch::Tensor hide_debug_read() {
  torch::Tensor out = torch::zeros({512, 16}, torch::dtype(torch::kInt64));
  cudaError_t e = cudaMemcpyFromSymbol(out.data_ptr<int64_t>(), hp_dbg, sizeof(unsigned long long) * 512 * 16);
  TORCH_CHECK(e == cudaSuccess, "hide_debug_read failed: ", cudaGetErrorString(e));
  return out;
}
void launch_cascade_select(torch::Tensor Planes, torch::Tensor Indices,
    torch::Tensor NextIdx, int S, int nblk, int topk, int win, int Skv) {
  TORCH_CHECK(topk + win <= 4096, "cascade select requires topk+win <= 4096");
  TORCH_CHECK((int64_t)nblk * S * (topk + win) <= Planes.numel(),
              "Planes too small for [nblk, S, topk+win]");
  TORCH_CHECK((int64_t)S * topk <= NextIdx.numel(),
              "NextIdx must be (S, topk) int32");
  cascade_select_kernel<<<dim3(S), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      Planes.data_ptr<float>(), Indices.data_ptr<int>(), NextIdx.data_ptr<int>(),
      S, nblk, topk, win, Skv);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "cascade select launch failed");
}
void launch_cascade_epilogue(torch::Tensor Planes, torch::Tensor Indices,
    torch::Tensor TapOut, int S, int nblk, int topk, int win, int Skv) {
  TORCH_CHECK((int64_t)nblk * S * (topk + win) <= Planes.numel(),
              "Planes too small for [nblk, S, topk+win]");
  TORCH_CHECK((int64_t)S * (topk + win) <= TapOut.numel(),
              "TapOut must be (S, topk+win) float32");
  cascade_epilogue_kernel<<<dim3(S), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      Planes.data_ptr<float>(), Indices.data_ptr<int>(), TapOut.data_ptr<float>(),
      S, nblk, topk, win, Skv, /*batched=*/0);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "cascade epilogue launch failed");
}
// Batched-decode variant: each of the S rows is an independent sequence at
// position Skv-1 (pairs with launch_fused_sparse_splitk_batch).
void launch_cascade_epilogue_batch(torch::Tensor Planes, torch::Tensor Indices,
    torch::Tensor TapOut, int S, int nblk, int topk, int win, int Skv) {
  TORCH_CHECK((int64_t)nblk * S * (topk + win) <= Planes.numel(),
              "Planes too small for [nblk, S, topk+win]");
  TORCH_CHECK((int64_t)S * (topk + win) <= TapOut.numel(),
              "TapOut must be (S, topk+win) float32");
  cascade_epilogue_kernel<<<dim3(S), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      Planes.data_ptr<float>(), Indices.data_ptr<int>(), TapOut.data_ptr<float>(),
      S, nblk, topk, win, Skv, /*batched=*/1);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "cascade epilogue batch launch failed");
}
static void launch_fused_impl(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexK, torch::Tensor IndexW, torch::Tensor IndexKScale,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlab,
    torch::Tensor SelHist, const float* tapw, int* next_idx, int* sel_counter,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int attend_k,
    int nsplit, double sm_scale, int with_select, int attn_on, int win,
    int64_t kv_stride, int64_t idx_k_stride, int64_t idx_scale_stride,
    // Paged/graph-capturable HIDE extras (all-null/zero = every legacy entry):
    const int* seqlens = nullptr, const int* page_table = nullptr,
    int64_t pt_stride = 0, int64_t page_bytes = 0, int64_t slab_stride = 0,
    int nsplit_attn = 0,
    int kv_fp8 = 0, int64_t kv_rows = 0, int sel_on = 0,
    unsigned* sel_scratch = nullptr, int* next_idx_out = nullptr, int sel_m = 0) {
  TORCH_CHECK(H % 16 == 0, "H must be a multiple of 16");
  TORCH_CHECK(nsplit_attn >= 0 && nsplit_attn <= nsplit,
              "nsplit_attn must be in [0, nsplit] (0 = every split attends)");
  TORCH_CHECK(attend_k >= 1 && attend_k <= topk,
              "attend_k must be in [1, topk]");
  const bool paged = (page_table != nullptr);
  if (paged) {
    // Paged HIDE: attention reads a COMPACT per-row buffer of the attended
    // topk rows (kv_stride >= topk*576); the indexer walks the page table, so
    // the dense IndexK/IndexKScale stride contract does not apply.
    TORCH_CHECK(with_select == 0 || with_select == 1,
                "paged launch supports with_select 0 (attention only) or 1 (fused indexer)");
    if (kv_fp8) {
      // fp8 pool path: KV is the serving latent pool [kv_rows, 656] (uint8/fp8 bytes),
      // Indices are raw KV slots; every attention block gathers 2*BNA rows.
      TORCH_CHECK(KV.element_size() == 1 && KV.dim() == 2 && KV.size(1) == 656,
                  "paged launch (kv_fp8): KV must be the [rows, 656] fp8-packed latent pool");
      TORCH_CHECK(kv_rows >= 1 && kv_rows <= KV.size(0), "paged launch (kv_fp8): bad kv_rows");
      TORCH_CHECK(!attn_on || nsplit_attn * 64 == topk,
                  "paged launch (kv_fp8): requires topk == 64 * nsplit_attn (two 32-key tiles per attention block)");
    } else {
      TORCH_CHECK(kv_stride >= (int64_t)topk * 576 && kv_stride % 8 == 0,
                  "paged launch: kv_stride must cover topk rows of 576 bf16 and stay 16-byte aligned");
      TORCH_CHECK(KV.numel() >= (int64_t)(S - 1) * kv_stride + (int64_t)topk * 576,
                  "paged launch: KV too small for the compact per-row attended rows");
    }
    TORCH_CHECK(seqlens != nullptr, "paged launch requires per-row SeqLens");
    TORCH_CHECK(pt_stride > 0 && (int64_t)pt_stride * 64 >= Skv,
                "paged launch: page table must cover the Skv capacity (pt_stride*64 >= Skv)");
    TORCH_CHECK(page_bytes == 64 * 128 + 64 * 4,
                "paged launch: index-K page must be [64 x 128 B keys][64 x fp32 scales]");
    TORCH_CHECK(slab_stride >= Skv, "paged launch: slab_stride must be >= Skv capacity");
  }
  if (kv_stride != 0 && !paged) {
    // Batched rows: the fp8 indexer path reads IndexK/IndexKScale with no
    // per-row stride, and the folded select's election counter layout is
    // single-sequence; only attention-only and cascade-tap modes are
    // row-independent.  576 = the kernel's compiled joint-KV row width
    // (kv_lora 512 + rope 64): a smaller stride would alias rows.
    // R2b: the fp8-indexer modes are batched-capable when per-row
    // IndexK/IndexKScale strides are supplied (the batch_idx entry); the
    // folded-select mode 4 keeps its single-sequence election layout.
    bool do_sel_batched = (with_select == 1 || with_select == 2);
    // union of the R2b batched-indexer entry (1/2) and the per-head exp-tap
    // plane (5, GLM batched tap) — both batched-capable, merged 2026-08-11.
    TORCH_CHECK(with_select == 0 || with_select == 3 || with_select == 5 || do_sel_batched,
                "batched launch (kv_stride>0) supports with_select 0, 1, 2, 3 or 5");
    if (do_sel_batched) {
      TORCH_CHECK(idx_k_stride > 0 && idx_scale_stride > 0,
                  "batched fused-indexer modes need per-row IndexK/IndexKScale "
                  "strides (use launch_fused_sparse_splitk_batch_idx)");
      // The indexer prefetch copies IndexK rows through int4 casts, so every
      // row base must stay 16-byte aligned (fp8 = 1 B/elem).
      TORCH_CHECK(idx_k_stride % 16 == 0,
                  "idx_k_stride must be a multiple of 16 fp8 elements");
      TORCH_CHECK(idx_k_stride >= (int64_t)Skv * 128,
                  "idx_k_stride must cover Skv rows of the 128-wide index keys");
      TORCH_CHECK(IndexK.numel() >= (int64_t)(S - 1) * idx_k_stride + (int64_t)Skv * 128,
                  "IndexK too small for the strided batch");
      TORCH_CHECK(idx_scale_stride >= (int64_t)Skv,
                  "idx_scale_stride must cover Skv per-key scales");
      TORCH_CHECK(IndexKScale.numel() >= (int64_t)(S - 1) * idx_scale_stride + (int64_t)Skv,
                  "IndexKScale too small for the strided batch");
    }
    TORCH_CHECK(kv_stride >= (int64_t)Skv * 576,
                "kv_stride must cover Skv rows of the 576-wide joint KV");
    // The kernel's __pipeline_memcpy_async copies read each row's KV base
    // (KV + s_i*kv_stride) through int4 casts, 16 bytes at a time, so the
    // per-row bf16 element stride must keep 16-byte alignment: a multiple
    // of 8 elements (8 * 2 B = 16 B).  A padded stride like Skv*576 + 4
    // passes the magnitude bound above but faults with misaligned-address
    // on every row past the first.
    TORCH_CHECK(kv_stride % 8 == 0,
                "kv_stride must be a multiple of 8 bf16 elements so every "
                "row base stays 16-byte aligned for the int4 async copies");
    // Footprint of ALL row bases, not just one row's width: row s reads
    // KV + s*kv_stride through KV + s*kv_stride + Skv*576, so a short
    // tensor (or an oversized stride) sends the last rows out of bounds
    // (CodeRabbit PR-29).
    TORCH_CHECK(KV.numel() >= (int64_t)(S - 1) * kv_stride + (int64_t)Skv * 576,
                "KV too small for the strided batch: needs at least "
                "(S-1)*kv_stride + Skv*576 elements");
  }
  if (with_select >= 3) {
    // cascade mode stores per-head-group tap planes into SelSlab viewed as
    // [nblk, S, topk+win]; the [S, Skv] allocation must be large enough, and
    // the attention warps must run (they produce the incumbent tap planes).
    TORCH_CHECK((int64_t)(H / 16) * S * (topk + win) <= SelSlab.numel(),
                "SelSlab too small for cascade planes [nblk, S, topk+win]");
    TORCH_CHECK(attn_on == 1, "with_select>=3 requires attn_on==1");
  }
  if (with_select == 1 || with_select == 2) {
    // Fused-indexer modes write full-prefix logits [S, Skv]; the slab was
    // previously the caller's unchecked contract (host-side asserts only).
    const int64_t slab_row = (slab_stride > 0) ? slab_stride : (int64_t)Skv;
    TORCH_CHECK(SelSlab.numel() >= (int64_t)S * slab_row,
                "SelSlab must hold [S, slab_row] logits for with_select 1/2");
  }
  int nblk = H / 16;
  TORCH_CHECK(IndexQ.size(-1) == 128, "index_head_dim must be 128");
  TORCH_CHECK(Hidx % 16 == 0, "index_n_heads must be a multiple of 16");
  // Padded strides (see kernel defines): KLD=584, SLD=36, PLD=40, IKLD=144.
  // No acc_o: the PV accumulator is register-resident. BNA=32 attention tile
  // + BNI=64 indexer tile => ~105 KB/block => 2 blocks/SM (launch_bounds 256,2).
  int base = (16*584 + 32*584) * 2 + (16*36) * 4 + (16*40) * 2 + (16+16+16+16) * 4;
  int idx_smem = Hidx * 4 + (Hidx * 144) * 1 + (2 * 64 * 144) * 1 + (64 * (Hidx + 2)) * 4;
  int smem_bytes = base + idx_smem + 32 * 4 + 256 * 4;   // + logit histogram bins
  static int configured = -1;
  if (configured != smem_bytes) {
    cudaError_t e = cudaFuncSetAttribute(fused_sparse_splitk_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    TORCH_CHECK(e == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(e));
    // ask for the max shared-memory carveout so two ~105 KB blocks co-reside
    e = cudaFuncSetAttribute(fused_sparse_splitk_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    TORCH_CHECK(e == cudaSuccess, "carveout attribute failed: ", cudaGetErrorString(e));
    int max_blocks = -1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks,
        fused_sparse_splitk_kernel, 256, smem_bytes);
    cudaFuncAttributes fattr;
    cudaFuncGetAttributes(&fattr, fused_sparse_splitk_kernel);
    printf("[occupancy] smem=%d B regs=%d blocks/SM=%d\n",
           smem_bytes, fattr.numRegs, max_blocks);
    configured = smem_bytes;
  }
  if (with_select == 2) {
    TORCH_CHECK(SelHist.numel() >= (int64_t)S * 256, "SelHist must be (S, 256) int32");
    // enforce the fresh-zero precondition: the epilogue accumulates into
    // SelHist, so stale counts would silently corrupt the histogram
    cudaMemsetAsync(SelHist.data_ptr<int>(), 0, sizeof(int) * (size_t)S * 256,
                    at::cuda::getCurrentCUDAStream());
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(S * nblk * nsplit);
  if (paged) {
    // Decoupled attention/indexer blocks, folded combine (see hide_paged_kernel).
    if (attn_on) {
      TORCH_CHECK(nsplit_attn >= 1, "paged launch: nsplit_attn must be >= 1 when attention is on");
      TORCH_CHECK(nsplit_attn <= 32, "paged launch: nsplit_attn must be <= 32 (combine weight table)");
      TORCH_CHECK(nblk <= 16, "paged launch: at most 16 head-groups (counter layout)");
      TORCH_CHECK(with_select == 0 || nsplit_attn < nsplit,
                  "paged launch: with_select=1 needs indexer blocks (nsplit_attn < nsplit)");
    }
    TORCH_CHECK(SelHist.numel() >= (int64_t)S * 256, "paged launch: SelHist [S,256] is the attention election counter");
    int attn_bytes = (16 * 584 + 2 * 32 * 584) * 2 + 2 * ((2 * 16 * 36) * 4 + (16 * 40) * 2 + 3 * 16 * 4 + 2 * 32 * 4);
    int idx_bytes = Hidx * 136 * 2 + 4 * 128 * 144 + Hidx * 4 + 4 * 128 * 4
                  + 2 * 128 * 4 + 1024 * 4 + 32 * 4 + 2048 * 4;   // + page ids, coarse histogram, select bookkeeping, slice logits
    // the wgmma operand region (1 KB alignment slack + 8 KB Q + 4 x 16 KB stages) must fit
    // inside the legacy [Q f16 | stages] span it aliases
    TORCH_CHECK(1024 + 8192 + 4 * 16384 <= Hidx * 136 * 2 + 4 * 128 * 144 + Hidx * 4 + 4 * 128 * 4,
                "wgmma operand region does not fit the indexer smem span for Hidx=", Hidx);
    TORCH_CHECK(Hidx <= 64, "paged launch: Hidx must be <= 64");
    int hp_smem = attn_bytes > idx_bytes ? attn_bytes : idx_bytes;
    static int hp_configured = -1;
    static int hp_capacity = 0;   // co-resident blocks the device can hold at this smem
    if (with_select == 1) {
      const int n_idx_row = nblk * (nsplit - (attn_on ? nsplit_attn : 0));
      TORCH_CHECK(n_idx_row > 0, "paged launch: with_select=1 needs indexer blocks");
      const int64_t nchunks = ((int64_t)Skv + 15) / 16;
      const int64_t my_chunks = (nchunks + n_idx_row - 1) / n_idx_row;   // 16-key interleaved slices
      TORCH_CHECK(my_chunks <= 2 * 128,
                  "paged launch: indexer slice too long for the page-id cache (raise nsplit)");
      if (sel_on) {
        TORCH_CHECK(sel_on == 1 || sel_on == 2, "paged launch: sel_on must be 0, 1 (exact) or 2 (approx)");
        TORCH_CHECK(sel_scratch != nullptr && next_idx_out != nullptr, "paged launch: sel_on needs SelScratch + NextIdx");
        TORCH_CHECK(topk <= 2048, "paged launch: in-kernel select supports topk <= 2048");
        if (sel_on == 2) {
          TORCH_CHECK(sel_m >= 1 && sel_m <= 2048 && (int64_t)n_idx_row * sel_m <= 8192 && (int64_t)n_idx_row * sel_m >= topk + sel_m
                      && ((int64_t)n_idx_row * sel_m) % 256 == 0,
                      "paged launch (approx): need topk + m <= n_idx * m <= 8192, a multiple of 256; got n_idx=", n_idx_row, " m=", sel_m);
          TORCH_CHECK(my_chunks * 16 <= 2048, "paged launch (approx): slice must fit the smem logit buffer (raise nsplit)");
        }
      }
    }
    if (hp_configured != hp_smem) {
      cudaError_t e = cudaFuncSetAttribute(hide_paged_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize, hp_smem);
      TORCH_CHECK(e == cudaSuccess, "cudaFuncSetAttribute(hide_paged) failed: ", cudaGetErrorString(e));
      e = cudaFuncSetAttribute(hide_paged_kernel,
          cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
      TORCH_CHECK(e == cudaSuccess, "carveout attribute (hide_paged) failed: ", cudaGetErrorString(e));
      int max_blocks = -1;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, hide_paged_kernel, 256, hp_smem);
      cudaFuncAttributes fattr;
      cudaFuncGetAttributes(&fattr, hide_paged_kernel);
      int dev = -1, n_sms = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev);
      hp_capacity = max_blocks * n_sms;
      printf("[occupancy hide_paged] smem=%d B (attn %d / idx %d) regs=%d blocks/SM=%d sms=%d capacity=%d\n",
             hp_smem, attn_bytes, idx_bytes, fattr.numRegs, max_blocks, n_sms, hp_capacity);
      hp_configured = hp_smem;
    }
    if (sel_on == 1) {
      // A row's indexer blocks barrier on each other: they must all be co-resident
      // (in-order dispatch + this capacity bound => no deadlock). The combine blocks
      // of the same row may spin alongside them.
      const int n_idx_row = nblk * (nsplit - (attn_on ? nsplit_attn : 0));
      TORCH_CHECK(n_idx_row + nblk * 8 <= hp_capacity,
                  "paged launch (sel_on): a row's indexer + combine blocks (", n_idx_row + nblk * 8,
                  ") exceed the co-resident capacity (", hp_capacity, "); lower nsplit");
    }
    dim3 hp_grid(S * nblk * nsplit + (attn_on ? S * nblk * 8 : 0));   // + HP_NCOMB combine blocks
    hide_paged_kernel<<<hp_grid, 256, hp_smem, stream>>>(
        (const __nv_bfloat16*)Q.data_ptr(), (const __nv_bfloat16*)KV.data_ptr(),
        Indices.data_ptr<int>(), (const __nv_fp8_e4m3*)IndexQ.data_ptr(),
        (const unsigned char*)IndexK.data_ptr(), IndexW.data_ptr<float>(),
        PartAcc.data_ptr<float>(), PartM.data_ptr<float>(), PartL.data_ptr<float>(),
        SelSlab.data_ptr<float>(), SelHist.data_ptr<int>(), (__nv_bfloat16*)Out.data_ptr(),
        S, Skv, H, Hidx, topk, nsplit, (float)sm_scale, with_select, attn_on, (long)kv_stride,
        seqlens, page_table, (long)pt_stride, (long)page_bytes, (long)slab_stride, nsplit_attn,
        kv_fp8, (long)kv_rows, sel_on, sel_scratch, next_idx_out, sel_m);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "hide_paged kernel launch failed");
    return;
  }
  fused_sparse_splitk_kernel<<<grid, 256, smem_bytes, stream>>>(
      (const __nv_bfloat16*)Q.data_ptr(), (const __nv_bfloat16*)KV.data_ptr(),
      Indices.data_ptr<int>(), (const __nv_fp8_e4m3*)IndexQ.data_ptr(),
      (const __nv_fp8_e4m3*)IndexK.data_ptr(), IndexW.data_ptr<float>(),
      // paged mode passes the uint8 page buffer here (scales live in-page)
      (const float*)IndexKScale.data_ptr(), tapw,
      PartAcc.data_ptr<float>(), PartM.data_ptr<float>(), PartL.data_ptr<float>(),
      SelSlab.data_ptr<float>(), SelHist.data_ptr<int>(), next_idx, sel_counter,
      S, Skv, H, Hidx, topk, attend_k, nsplit,
      (float)sm_scale, with_select, attn_on, win, (long)kv_stride,
      (long)idx_k_stride, (long)idx_scale_stride,
      seqlens, page_table, (long)pt_stride, (long)page_bytes, (long)slab_stride,
      nsplit_attn);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "splitk kernel launch failed");
  dim3 cgrid(S * nblk);
  combine_kernel<<<cgrid, 256, 0, stream>>>(PartAcc.data_ptr<float>(), PartM.data_ptr<float>(),
      PartL.data_ptr<float>(), (__nv_bfloat16*)Out.data_ptr(), S, H, nsplit,
      (nsplit_attn > 0) ? nsplit_attn : nsplit);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "combine kernel launch failed");
}
void launch_fused_sparse_splitk(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexK, torch::Tensor IndexW, torch::Tensor IndexKScale,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlab,
    torch::Tensor SelHist,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int nsplit,
    double sm_scale, int with_select, int attn_on, int win) {
  TORCH_CHECK(with_select != 4,
              "with_select==4 requires the selectfold entry point (NextIdx + counter)");
  // Mode 5 writes PER-HEAD planes [H, S, topk+win] — 16x the mode-3 plane
  // footprint launch_fused_impl validates SelSlab against — so a generic
  // launch would pass the check yet scribble past a mode-3-sized slab.
  TORCH_CHECK(with_select != 5,
              "with_select==5 requires the exptap entry point "
              "(launch_fused_sparse_splitk_exptap validates the per-head "
              "SelSlabH sizing)");
  launch_fused_impl(Q, KV, Indices, IndexQ, IndexK, IndexW, IndexKScale,
                    PartAcc, PartM, PartL, SelSlab, SelHist,
                    nullptr, nullptr, nullptr, Out, S, Skv, H, Hidx, topk,
                    /*attend_k=*/topk, nsplit,
                    sm_scale, with_select, attn_on, win, /*kv_stride=*/0,
                    /*idx_k_stride=*/0, /*idx_scale_stride=*/0);
}
// Cascade with distilled per-head tap weights: identical to with_select==3
// via the plain entry, except the plane accumulation weights each head's
// ReLU'd score by TapW[h] (the closed-form ridge fit of docs/results.md).
void launch_fused_sparse_splitk_tapw(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexK, torch::Tensor IndexW, torch::Tensor IndexKScale,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlab,
    torch::Tensor SelHist, torch::Tensor TapW,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int nsplit,
    double sm_scale, int win) {
  TORCH_CHECK(TapW.scalar_type() == torch::kFloat32, "TapW must be float32");
  TORCH_CHECK(TapW.numel() >= H, "TapW must hold one weight per attention head");
  TORCH_CHECK(TapW.is_contiguous(), "TapW must be contiguous");
  launch_fused_impl(Q, KV, Indices, IndexQ, IndexK, IndexW, IndexKScale,
                    PartAcc, PartM, PartL, SelSlab, SelHist,
                    TapW.data_ptr<float>(), nullptr, nullptr,
                    Out, S, Skv, H, Hidx, topk, /*attend_k=*/topk, nsplit,
                    sm_scale, /*with_select=*/3, /*attn_on=*/1, win, /*kv_stride=*/0,
                    /*idx_k_stride=*/0, /*idx_scale_stride=*/0);
}
// Pool/attend decoupling: Indices holds the score-sorted candidate POOL
// (`topk` slots); softmax/PV consume only the leading attend_k columns while
// the tap planes cover the full pool plus the admission window.  The pool's
// tail thus rides the attention warps' QK tiles as score-only keys — the
// same mechanism the admission window uses, generalized from recency-picked
// candidates to tap-retained ones.  TapW is required (pass ones for the
// uniform tap; a positive rescale never changes the selected set).
void launch_fused_sparse_splitk_pool(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexK, torch::Tensor IndexW, torch::Tensor IndexKScale,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlab,
    torch::Tensor SelHist, torch::Tensor TapW,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int attend_k,
    int nsplit, double sm_scale, int win) {
  TORCH_CHECK(TapW.scalar_type() == torch::kFloat32, "TapW must be float32");
  TORCH_CHECK(TapW.numel() >= H, "TapW must hold one weight per attention head");
  TORCH_CHECK(TapW.is_contiguous(), "TapW must be contiguous");
  launch_fused_impl(Q, KV, Indices, IndexQ, IndexK, IndexW, IndexKScale,
                    PartAcc, PartM, PartL, SelSlab, SelHist,
                    TapW.data_ptr<float>(), nullptr, nullptr,
                    Out, S, Skv, H, Hidx, topk, attend_k, nsplit,
                    sm_scale, /*with_select=*/3, /*attn_on=*/1, win, /*kv_stride=*/0,
                    /*idx_k_stride=*/0, /*idx_scale_stride=*/0);
}
// Batched decode: the S query rows are S INDEPENDENT sequences, each reading
// its own KV at KV + row*kv_stride (elements) — the [B, T, 576] joint cache
// with kv_stride = T*576 — all decoding at the same position Skv-1.  Per-row
// arithmetic is identical to a legacy launch on that row's KV slice, so the
// batched output is bitwise-equal to S single-row launches.  with_select 0
// (refresh/final layers: attention only) or 3 (cascade tap planes); TapW is
// required for mode 3 (ones = uniform) and ignored for mode 0.
void launch_fused_sparse_splitk_batch(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexK, torch::Tensor IndexW, torch::Tensor IndexKScale,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlab,
    torch::Tensor SelHist, torch::Tensor TapW,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int attend_k,
    int nsplit, double sm_scale, int with_select, int win, int64_t kv_stride) {
  TORCH_CHECK(kv_stride > 0, "batched entry requires kv_stride > 0");
  // Same footgun as the single-row entry: mode 5 writes PER-HEAD planes
  // [H, S, topk+win], 16x the plane footprint launch_fused_impl validates
  // SelSlab against — the exptap_batch entry checks the per-head sizing.
  TORCH_CHECK(with_select != 5,
              "with_select==5 requires the exptap_batch entry point "
              "(launch_fused_sparse_splitk_exptap_batch validates the "
              "per-head SelSlabH sizing)");
  const float* tapw = nullptr;
  if (with_select == 3) {
    TORCH_CHECK(TapW.scalar_type() == torch::kFloat32, "TapW must be float32");
    TORCH_CHECK(TapW.numel() >= H, "TapW must hold one weight per attention head");
    TORCH_CHECK(TapW.is_contiguous(), "TapW must be contiguous");
    tapw = TapW.data_ptr<float>();
  }
  launch_fused_impl(Q, KV, Indices, IndexQ, IndexK, IndexW, IndexKScale,
                    PartAcc, PartM, PartL, SelSlab, SelHist,
                    tapw, nullptr, nullptr,
                    Out, S, Skv, H, Hidx, topk, attend_k, nsplit,
                    sm_scale, with_select, /*attn_on=*/1, win, kv_stride,
                    /*idx_k_stride=*/0, /*idx_scale_stride=*/0);
}

// R2b batched fused-indexer entry: with_select 1/2 over independent batch
// rows.  Per-row bases mirror the KV convention: row s reads
// IndexK + s*idx_k_stride (fp8 elements, 16-byte-aligned stride) and
// IndexKScale + s*idx_scale_stride (fp32 elements); SelSlab receives the
// full-prefix logits packed [S, Skv].  Attention runs over Indices exactly
// as the plain batch entry (attend_k == topk); modes 0/3 keep their
// existing entries.  SGLANG PORT NOTE (docs/track_source_design.md, section 7):
// the contract assumes per-request CONTIGUOUS index-key slabs; a paged
// layout needs either a contiguous per-request idx-K pool or a page-table
// walking variant.
void launch_fused_sparse_splitk_batch_idx(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexK, torch::Tensor IndexW, torch::Tensor IndexKScale,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlab,
    torch::Tensor SelHist,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int nsplit,
    double sm_scale, int with_select, int attn_on, int win, int64_t kv_stride,
    int64_t idx_k_stride, int64_t idx_scale_stride) {
  TORCH_CHECK(with_select == 1 || with_select == 2,
              "batch_idx is the batched fused-indexer entry (with_select 1/2); "
              "modes 0/3 use the plain batch entry");
  launch_fused_impl(Q, KV, Indices, IndexQ, IndexK, IndexW, IndexKScale,
                    PartAcc, PartM, PartL, SelSlab, SelHist,
                    nullptr, nullptr, nullptr, Out, S, Skv, H, Hidx, topk,
                    /*attend_k=*/topk, nsplit,
                    sm_scale, with_select, attn_on, win, kv_stride,
                    idx_k_stride, idx_scale_stride);
}
// Paged, graph-capturable HIDE entry (serving): per-row device SeqLens replace
// the host-int prefix length, and the indexer scans the PAGED serving index-K
// cache through PageTable [S, pt_stride] (64-token pages of
// [64 x 128 B fp8 keys][64 x fp32 scales], page_bytes = 8448). Attention runs
// over the compact per-row attended rows in KV (kv_stride >= topk*576).
// SelSlab is [S, slab_stride] (positions >= the row's length are NOT written;
// the caller pre-fills -inf). Skv is the CAPACITY (== causal sentinel).
// with_select 1 = fused indexer+attention (HIDE); 0 / attn_on flags let a
// gate time the two halves separately with identical addressing.
void launch_fused_sparse_splitk_paged(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexKPaged, torch::Tensor IndexW,
    torch::Tensor SeqLens, torch::Tensor PageTable,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlab,
    torch::Tensor SelHist,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int nsplit,
    double sm_scale, int with_select, int attn_on, int64_t kv_stride,
    int64_t pt_stride, int64_t page_bytes, int64_t slab_stride, int nsplit_attn,
    int kv_fp8, int64_t kv_rows, int sel_on, torch::Tensor SelScratch, torch::Tensor NextIdx, int sel_m) {
  TORCH_CHECK(SeqLens.scalar_type() == torch::kInt32 && SeqLens.is_contiguous()
              && SeqLens.numel() >= S, "SeqLens must be contiguous int32 [S]");
  unsigned* sel_scratch = nullptr;
  int* next_idx_out = nullptr;
  if (sel_on) {
    TORCH_CHECK(SelScratch.scalar_type() == torch::kInt32 && SelScratch.is_contiguous()
                && SelScratch.numel() >= (int64_t)S * (1056 + 2 * 2048 + 2 * 8192),
                "SelScratch must be contiguous int32 [S, 21536] (zero at first launch; self-resetting)");
    TORCH_CHECK(NextIdx.scalar_type() == torch::kInt32 && NextIdx.is_contiguous()
                && NextIdx.numel() >= (int64_t)S * topk, "NextIdx must be contiguous int32 [S, topk]");
    TORCH_CHECK(Indices.scalar_type() == torch::kInt32, "Indices must be int32");
    sel_scratch = (unsigned*)SelScratch.data_ptr<int>();
    next_idx_out = NextIdx.data_ptr<int>();
  }
  TORCH_CHECK(PageTable.scalar_type() == torch::kInt32 && PageTable.is_contiguous()
              && PageTable.numel() >= (int64_t)S * pt_stride,
              "PageTable must be contiguous int32 [S, pt_stride]");
  TORCH_CHECK(IndexKPaged.scalar_type() == torch::kUInt8 && IndexKPaged.dim() == 2
              && IndexKPaged.size(1) == page_bytes,
              "IndexKPaged must be the uint8 [num_pages, page_bytes] serving index-K buffer");
  launch_fused_impl(Q, KV, Indices, IndexQ, IndexKPaged, IndexW, /*IndexKScale=*/IndexKPaged,
                    PartAcc, PartM, PartL, SelSlab, SelHist,
                    nullptr, nullptr, nullptr, Out, S, Skv, H, Hidx, topk,
                    /*attend_k=*/topk, nsplit,
                    sm_scale, with_select, attn_on, /*win=*/0, kv_stride,
                    /*idx_k_stride=*/0, /*idx_scale_stride=*/0,
                    SeqLens.data_ptr<int>(), PageTable.data_ptr<int>(),
                    pt_stride, page_bytes, slab_stride, nsplit_attn,
                    kv_fp8, kv_rows, sel_on, sel_scratch, next_idx_out, sel_m);
}
// with_select==4: single-launch cascade — fused attention + tap planes + the
// radix select folded into the same kernel via last-arriver election. The
// per-row counter is zeroed here so every launch (incl. graph replays, which
// capture the memset) starts a fresh election.
void launch_fused_sparse_splitk_selectfold(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexK, torch::Tensor IndexW, torch::Tensor IndexKScale,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlab,
    torch::Tensor SelHist, torch::Tensor NextIdx, torch::Tensor SelCounter,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int nsplit,
    double sm_scale, int win) {
  TORCH_CHECK(topk + win <= 4096, "folded select requires topk+win <= 4096");
  TORCH_CHECK((int64_t)S * topk <= NextIdx.numel(), "NextIdx must be (S, topk) int32");
  TORCH_CHECK(NextIdx.scalar_type() == torch::kInt32, "NextIdx must be int32");
  TORCH_CHECK(SelCounter.numel() >= S, "SelCounter must be (S,) int32");
  TORCH_CHECK(SelCounter.scalar_type() == torch::kInt32, "SelCounter must be int32");
  cudaMemsetAsync(SelCounter.data_ptr<int>(), 0, sizeof(int) * (size_t)S,
                  at::cuda::getCurrentCUDAStream());
  launch_fused_impl(Q, KV, Indices, IndexQ, IndexK, IndexW, IndexKScale,
                    PartAcc, PartM, PartL, SelSlab, SelHist,
                    nullptr, NextIdx.data_ptr<int>(), SelCounter.data_ptr<int>(),
                    Out, S, Skv, H, Hidx, topk, /*attend_k=*/topk, nsplit,
                    sm_scale, /*with_select=*/4, /*attn_on=*/1, win, /*kv_stride=*/0,
                    /*idx_k_stride=*/0, /*idx_scale_stride=*/0);
}
// Exp-tap cascade (with_select==5, the fused v4_probs scoring): attention is
// byte-identical to mode 3, but the plane-write sites store PER-HEAD raw
// log2-domain scores into SelSlabH ([H, S, topk+win], threaded through the
// kernel's SelSlab pointer — the same mode-dependent reinterpretation modes
// 3/4 already use for [nblk, S, topk+win]).  launch_cascade_exp_epilogue
// turns them into the normalized head-summed probabilities.  TapW is
// structurally absent: the verified v4 arm is uniform, so this entry takes
// no weight tensor and passes nullptr (the mode-5 fail-fast the spec asks
// for).  Single-sequence only (kv_stride hardwired 0); batched decode goes
// through launch_fused_sparse_splitk_exptap_batch below.
void launch_fused_sparse_splitk_exptap(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexK, torch::Tensor IndexW, torch::Tensor IndexKScale,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlabH,
    torch::Tensor SelHist,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int nsplit,
    double sm_scale, int win) {
  TORCH_CHECK(SelSlabH.scalar_type() == torch::kFloat32, "SelSlabH must be float32");
  TORCH_CHECK((int64_t)H * S * (topk + win) <= SelSlabH.numel(),
              "SelSlabH too small for per-head exp-tap planes [H, S, topk+win]");
  launch_fused_impl(Q, KV, Indices, IndexQ, IndexK, IndexW, IndexKScale,
                    PartAcc, PartM, PartL, SelSlabH, SelHist,
                    /*tapw=*/nullptr, nullptr, nullptr,
                    Out, S, Skv, H, Hidx, topk, /*attend_k=*/topk, nsplit,
                    sm_scale, /*with_select=*/5, /*attn_on=*/1, win, /*kv_stride=*/0,
                    /*idx_k_stride=*/0, /*idx_scale_stride=*/0);
}
// Batched exp-tap cascade: the S query rows are S INDEPENDENT sequences (row
// s reads KV + s*kv_stride, every row decodes at position Skv-1 — identical
// semantics to launch_fused_sparse_splitk_batch), with the mode-5 per-head
// plane writes of the single-sequence exptap entry.  Per-row arithmetic is
// unchanged, so a batched launch is bitwise-equal to S single-row exptap
// launches on the row slices (certified by validate_exptap_batched).
void launch_fused_sparse_splitk_exptap_batch(torch::Tensor Q, torch::Tensor KV, torch::Tensor Indices,
    torch::Tensor IndexQ, torch::Tensor IndexK, torch::Tensor IndexW, torch::Tensor IndexKScale,
    torch::Tensor PartAcc, torch::Tensor PartM, torch::Tensor PartL, torch::Tensor SelSlabH,
    torch::Tensor SelHist,
    torch::Tensor Out, int S, int Skv, int H, int Hidx, int topk, int nsplit,
    double sm_scale, int win, int64_t kv_stride) {
  TORCH_CHECK(kv_stride > 0, "batched exptap entry requires kv_stride > 0");
  TORCH_CHECK(SelSlabH.scalar_type() == torch::kFloat32, "SelSlabH must be float32");
  TORCH_CHECK((int64_t)H * S * (topk + win) <= SelSlabH.numel(),
              "SelSlabH too small for per-head exp-tap planes [H, S, topk+win]");
  launch_fused_impl(Q, KV, Indices, IndexQ, IndexK, IndexW, IndexKScale,
                    PartAcc, PartM, PartL, SelSlabH, SelHist,
                    /*tapw=*/nullptr, nullptr, nullptr,
                    Out, S, Skv, H, Hidx, topk, /*attend_k=*/topk, nsplit,
                    sm_scale, /*with_select=*/5, /*attn_on=*/1, win, kv_stride,
                    /*idx_k_stride=*/0, /*idx_scale_stride=*/0);
}
static void launch_cascade_exp_epilogue_impl(torch::Tensor PlanesH, torch::Tensor PartM,
    torch::Tensor PartL, torch::Tensor Indices, torch::Tensor TapOut, torch::Tensor ExpStats,
    int S, int nblk, int topk, int win, int Skv, int nsplit, int batched) {
  // EPI_MAXH bound of the kernel's static per-head shared buffers.
  TORCH_CHECK(nblk * 16 <= 256, "exp epilogue supports at most 256 local heads");
  TORCH_CHECK(nsplit > 0, "nsplit must be positive");
  TORCH_CHECK((int64_t)nblk * 16 * S * (topk + win) <= PlanesH.numel(),
              "PlanesH too small for per-head planes [nblk*16, S, topk+win]");
  TORCH_CHECK((int64_t)S * nblk * nsplit * 16 <= PartM.numel(),
              "PartM must cover [S*nblk*nsplit, 16]");
  TORCH_CHECK((int64_t)S * nblk * nsplit * 16 <= PartL.numel(),
              "PartL must cover [S*nblk*nsplit, 16]");
  TORCH_CHECK((int64_t)S * (topk + win) <= TapOut.numel(),
              "TapOut must be (S, topk+win) float32");
  TORCH_CHECK(ExpStats.scalar_type() == torch::kFloat32, "ExpStats must be float32");
  TORCH_CHECK((int64_t)S * 2 * nblk * 16 <= ExpStats.numel(),
              "ExpStats must be (S, 2, nblk*16) float32");
  cascade_exp_epilogue_kernel<<<dim3(S), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      PlanesH.data_ptr<float>(), PartM.data_ptr<float>(), PartL.data_ptr<float>(),
      Indices.data_ptr<int>(), TapOut.data_ptr<float>(), ExpStats.data_ptr<float>(),
      S, nblk, topk, win, Skv, nsplit, batched);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "cascade exp epilogue launch failed");
}
void launch_cascade_exp_epilogue(torch::Tensor PlanesH, torch::Tensor PartM, torch::Tensor PartL,
    torch::Tensor Indices, torch::Tensor TapOut, torch::Tensor ExpStats, int S, int nblk,
    int topk, int win, int Skv, int nsplit) {
  launch_cascade_exp_epilogue_impl(PlanesH, PartM, PartL, Indices, TapOut, ExpStats,
                                   S, nblk, topk, win, Skv, nsplit, /*batched=*/0);
}
// Batched-decode variant: each of the S rows is an independent sequence at
// position Skv-1 (pairs with launch_fused_sparse_splitk_exptap_batch).
void launch_cascade_exp_epilogue_batch(torch::Tensor PlanesH, torch::Tensor PartM, torch::Tensor PartL,
    torch::Tensor Indices, torch::Tensor TapOut, torch::Tensor ExpStats, int S, int nblk,
    int topk, int win, int Skv, int nsplit) {
  launch_cascade_exp_epilogue_impl(PlanesH, PartM, PartL, Indices, TapOut, ExpStats,
                                   S, nblk, topk, win, Skv, nsplit, /*batched=*/1);
}
"""


def build():
    return load_inline(
        name="fused_sparse_splitk_bal_ext",
        cpp_sources=[CPP_SRC],
        cuda_sources=[CUDA_SRC + "\n" + LAUNCH_SRC],
        functions=["launch_fused_sparse_splitk", "launch_fused_sparse_splitk_selectfold",
                   "launch_fused_sparse_splitk_tapw", "launch_fused_sparse_splitk_pool",
                   "launch_fused_sparse_splitk_batch",
                   "launch_fused_sparse_splitk_batch_idx",
                   "launch_fused_sparse_splitk_paged",
                   "launch_fused_sparse_splitk_exptap",
                   "launch_fused_sparse_splitk_exptap_batch",
                   "launch_cascade_epilogue", "launch_cascade_epilogue_batch",
                   "launch_cascade_exp_epilogue",
                   "launch_cascade_exp_epilogue_batch",
                   "launch_cascade_select", "hide_debug_enable", "hide_debug_read", "hide_diag_mode"],
        # sm_90a: the wgmma scan path is an arch-specific Hopper feature
        extra_cuda_cflags=["-O3", "-gencode", "arch=compute_90a,code=sm_90a"],
        verbose=False,
    )


def corr_diff(a, b):
    a, b = a.double(), b.double()
    n = (a * a + b * b).sum()
    return 0.0 if n == 0 else float(1.0 - 2 * (a * b).sum() / n)


def ref_sparse_mla(q, kv, indices, sm_scale, dev):
    _, S, H, dqk = q.shape
    skv = kv.shape[1]
    dv = 512
    out = torch.zeros(1, S, H, dv, dtype=torch.float32, device=dev)
    kvf = kv[0, :, 0, :].float()
    for s in range(S):
        q_pos = (skv - S) + s
        idx = indices[s]
        sel = idx[idx <= q_pos].long()
        if sel.numel() == 0:
            continue
        k = kvf[sel]
        scores = (q[0, s].float() @ k.t()) * sm_scale
        p = torch.softmax(scores, dim=-1)
        out[0, s] = p @ k[:, :dv]
    return out


def quantize_e4m3_rowwise(x):
    """Per-row (last-dim) symmetric e4m3 quantization -> (fp8 tensor, scale)."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    scale = amax / 448.0
    q = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale.squeeze(-1)


def make_fp8_indexer_inputs(S, Skv, Hidx, didx, dev):
    """Build the fp8 indexer operands the kernel consumes, plus a faithful fp32
    reference computed from the *same* fp8 values (so only the in-kernel f32
    accumulation can differ). Row scales fold into ``wp`` (q) and ``sk`` (k)."""
    iq = torch.randn(S, Hidx, didx, device=dev)
    ik = torch.randn(Skv, didx, device=dev)
    w = torch.randn(S, Hidx, device=dev).abs()
    iqf8, sq = quantize_e4m3_rowwise(iq)        # [S,Hidx,didx], [S,Hidx]
    ikf8, sk = quantize_e4m3_rowwise(ik)        # [Skv,didx],   [Skv]
    wp = (sq * w).contiguous()                  # [S,Hidx]  q-scale folded in
    return iqf8.contiguous(), ikf8.contiguous(), wp, sk.contiguous()


def ref_indexer_logits_fp8(iqf8, ikf8, wp, sk):
    bil = torch.einsum("shd,gd->shg", iqf8.float(), ikf8.float())   # [S,Hidx,Skv]
    return (bil.relu() * wp.unsqueeze(-1)).sum(1) * sk.view(1, -1)  # [S,Skv]


def make_indices(S, skv, topk, dev):
    indices = torch.full((S, topk), skv, dtype=torch.int32, device=dev)
    for s in range(S):
        q_pos = (skv - S) + s
        prefix = q_pos + 1
        take = min(topk, prefix)
        indices[s, :take] = torch.randperm(prefix, device=dev)[:take].to(torch.int32)
    return indices


def alloc_part(S, H, nsplit, dev):
    nb = S * (H // 16) * nsplit
    return (torch.zeros(nb, 16, 512, dtype=torch.float32, device=dev),
            torch.zeros(nb, 16, dtype=torch.float32, device=dev),
            torch.zeros(nb, 16, dtype=torch.float32, device=dev))


def _time_cuda(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters


def measure_cascade_parts(ext, dev, windows, nsplit, H, dqk, Hidx, didx, topk, S=1):
    """Context-independent components of the tap-cascade decode path.

    Under the cascade the per-layer selection becomes: a tap re-rank of the
    attended top-k (rides the kernel's existing epilogue — the same placement
    the with_select=2 histogram uses, measured free in docs/results.md §15),
    an fp8 indexer over only the W admission-window keys, and a top-k over
    (topk + W) candidates.  Neither the admission indexer nor the candidate
    top-k depends on context length — that is the point: the full-prefix
    indexer scan and the full-slab selector drop out of steady state.
    """
    from lagged_indexer_dsa.tilelang_dsa.topk_selector import tl_topk

    sm_scale = dqk ** -0.5
    parts: dict[int, dict[str, float]] = {}
    for W in windows:
        q = torch.randn(1, S, H, dqk, dtype=torch.bfloat16, device=dev)
        kv = torch.randn(1, W, 1, dqk, dtype=torch.bfloat16, device=dev)
        iqf8, ikf8, wp, sk = make_fp8_indexer_inputs(S, W, Hidx, didx, dev)
        indices = make_indices(S, W, topk, dev)
        out = torch.zeros(1, S, H, 512, dtype=torch.bfloat16, device=dev)
        slab = torch.zeros(S, W, dtype=torch.float32, device=dev)
        hist = torch.zeros(S, 256, dtype=torch.int32, device=dev)
        pa, pm, pl = alloc_part(S, H, nsplit, dev)
        adm = _time_cuda(lambda: ext.launch_fused_sparse_splitk(
            q, kv, indices, iqf8, ikf8, wp, sk, pa, pm, pl, slab, hist, out,
            S, W, H, Hidx, topk, nsplit, sm_scale, 1, 0, 0))

        cand = topk + W
        logits = torch.randn(S, cand, device=dev)
        starts = torch.zeros(S, dtype=torch.int32, device=dev)
        ends = torch.full((S,), cand, dtype=torch.int32, device=dev)
        tk_tl = _time_cuda(lambda: tl_topk(logits, starts, ends, topk, strict=False))
        tk_torch = _time_cuda(lambda: torch.topk(logits, topk, dim=-1))
        parts[W] = {
            "admission_ms": adm,
            "topk_tl_ms": tk_tl,
            "topk_torch_ms": tk_torch,
        }
        logger.info(
            "cascade parts W=%-5d (B=%d): admission idx=%.4f ms | "
            "cand top-k tl=%.4f torch=%.4f ms",
            W, S, adm, tk_tl, tk_torch,
        )
    return parts


def validate(ext, H, dqk, Hidx, didx, S, Skv, nsplit, dev, tag):
    topk = 2048
    q = torch.randn(1, S, H, dqk, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(1, Skv, 1, dqk, dtype=torch.bfloat16, device=dev)
    iqf8, ikf8, wp, sk = make_fp8_indexer_inputs(S, Skv, Hidx, didx, dev)
    indices = make_indices(S, Skv, topk, dev)
    out = torch.zeros(1, S, H, 512, dtype=torch.bfloat16, device=dev)
    # Flat slab covers legacy [S, Skv] logits and cascade planes
    # [nblk, S, topk+win] (the latter dominates at small Skv).
    val_win = 128
    slab_flat = torch.zeros(
        max(S * Skv, (H // 16) * S * (topk + val_win)),
        dtype=torch.float32, device=dev)
    slab = slab_flat[: S * Skv].view(S, Skv)
    pa, pm, pl = alloc_part(S, H, nsplit, dev)
    sm_scale = dqk ** -0.5

    hist = torch.zeros(S, 256, dtype=torch.int32, device=dev)
    ext.launch_fused_sparse_splitk(q, kv, indices, iqf8, ikf8, wp, sk, pa, pm, pl, slab_flat, hist, out,
                                   S, Skv, H, Hidx, topk, nsplit, sm_scale, 1, 1, 0)
    torch.cuda.synchronize()

    ref_out = ref_sparse_mla(q, kv, indices, sm_scale, dev)
    a_corr = corr_diff(out.float(), ref_out)
    a_rel = float((out.float() - ref_out).abs().max() / (ref_out.abs().max() + 1e-6))

    ref_logits = ref_indexer_logits_fp8(iqf8, ikf8, wp, sk)
    qpos = torch.arange(S, device=dev).view(S, 1) + (Skv - S)
    causal = torch.arange(Skv, device=dev).view(1, Skv) <= qpos
    s_corr = corr_diff(slab[causal], ref_logits[causal])
    s_rel = float((slab[causal] - ref_logits[causal]).abs().max() / (ref_logits[causal].abs().max() + 1e-6))

    assert a_corr < 1e-3 and a_rel < 5e-2, f"[{tag}] attn corr={a_corr} rel={a_rel}"
    assert s_corr < 1e-3 and s_rel < 2e-2, f"[{tag}] logits corr={s_corr} rel={s_rel}"

    # with_select=2: histogram epilogue + tl_topk from-hist path
    from lagged_indexer_dsa.tilelang_dsa.topk_selector import tl_topk, torch_bucket16

    legacy_logits = slab.clone()
    slab.zero_()
    ext.launch_fused_sparse_splitk(q, kv, indices, iqf8, ikf8, wp, sk, pa, pm, pl, slab_flat, hist, out,
                                   S, Skv, H, Hidx, topk, nsplit, sm_scale, 2, 1, 0)
    torch.cuda.synchronize()
    assert bool((slab == legacy_logits).all()), f"[{tag}] with_select=2 changed the logits"

    buckets = torch.where(causal, torch_bucket16(slab), torch.full_like(slab, 256, dtype=torch.int32))
    hist_ref = torch.zeros(S, 257, dtype=torch.int32, device=dev)
    hist_ref.scatter_add_(1, buckets.long(), torch.ones_like(buckets, dtype=torch.int32))
    assert bool((hist == hist_ref[:, :256]).all()), f"[{tag}] fused histogram disagrees with reference"

    starts = torch.zeros(S, dtype=torch.int32, device=dev)
    ends = (qpos.view(-1) + 1).to(torch.int32)
    idx_l = tl_topk(slab, starts, ends, topk)
    idx_f = tl_topk(slab, starts, ends, topk, hist=hist)
    agree = float(torch.tensor([
        len(set(idx_l[i].tolist()) & set(idx_f[i].tolist())) / topk for i in range(S)
    ]).mean())
    assert agree >= 0.999, f"[{tag}] from-hist top-k disagrees: agreement={agree}"

    logger.info("[%-14s nsplit=%2d] sparse-attn 1-c=%.2e rel=%.2e | logit 1-c=%.2e rel=%.2e | hist+topk OK",
                tag, nsplit, a_corr, a_rel, s_corr, s_rel)

    # with_select=3 (tap cascade): attention output unchanged; SelSlab planes
    # hold Σ_h ReLU(q_h·k) per head-group over incumbent cols + admission
    # window; the host union top-k matches the tap.cascade_topk reference.
    from lagged_indexer_dsa.tap import cascade_topk

    win = val_win
    nblk = H // 16
    ext.launch_fused_sparse_splitk(q, kv, indices, iqf8, ikf8, wp, sk, pa, pm, pl, slab_flat, hist, out,
                                   S, Skv, H, Hidx, topk, nsplit, sm_scale, 1, 1, 0)
    torch.cuda.synchronize()
    out_mode1 = out.clone()
    out.zero_()
    slab_flat.zero_()
    ext.launch_fused_sparse_splitk(q, kv, indices, iqf8, ikf8, wp, sk, pa, pm, pl, slab_flat, hist, out,
                                   S, Skv, H, Hidx, topk, nsplit, sm_scale, 3, 1, win)
    torch.cuda.synchronize()
    a3_corr = corr_diff(out.float(), ref_out)
    assert a3_corr < 1e-3, f"[{tag}] cascade mode changed attention: 1-c={a3_corr}"
    assert bool((out == out_mode1).all()), f"[{tag}] mode 3 attention not bitwise-identical to mode 1"

    ntap = topk + win
    tap_planes = slab_flat[: nblk * S * ntap].view(nblk, S, ntap)
    tap_got = tap_planes.sum(0)  # [S, topk+win]

    qf = q[0].float()  # [S, H, dqk]
    qpos_v = qpos.view(-1)
    ref_tap = torch.zeros(S, ntap, device=dev)
    win_idx = torch.zeros(S, win, dtype=torch.long, device=dev)
    for s in range(S):
        p_end = int(qpos_v[s].item()) + 1
        w_lo = max(p_end - win, 0)
        inc = indices[s].long().clamp(max=Skv - 1)
        keys = torch.cat([inc, torch.arange(w_lo, w_lo + win, device=dev)])
        scores = torch.einsum("hd,kd->hk", qf[s], kv[0, keys, 0].float())
        valid = torch.cat([indices[s].long() <= qpos_v[s],
                           torch.arange(w_lo, w_lo + win, device=dev) < p_end])
        ref_tap[s] = scores.relu().sum(0) * valid.float()
        win_idx[s] = torch.arange(w_lo, w_lo + win, device=dev)
    t_corr = corr_diff(tap_got, ref_tap)
    assert t_corr < 1e-3, f"[{tag}] cascade tap logits disagree: 1-c={t_corr}"

    cand_idx = torch.cat([indices.long(), win_idx], dim=1).unsqueeze(0)
    got_next = cascade_topk(cand_idx, tap_got.unsqueeze(0), topk,
                            query_positions=qpos_v)
    ref_next = cascade_topk(cand_idx, ref_tap.unsqueeze(0), topk,
                            query_positions=qpos_v)
    agree3 = float(torch.tensor([
        len(set(got_next[0, i].tolist()) & set(ref_next[0, i].tolist())) / topk
        for i in range(S)
    ]).mean())
    assert agree3 >= 0.999, f"[{tag}] cascade next-indices disagree: agreement={agree3}"

    # Epilogue kernel: bit-exact vs a torch mirror that reproduces the
    # kernel's sequential plane-summation order (torch.sum's reduction order
    # is unspecified, so the mirror accumulates plane-by-plane), with
    # incumbent∩window duplicates marked -inf.
    tap_out = torch.zeros(S, ntap, dtype=torch.float32, device=dev)
    ext.launch_cascade_epilogue(slab_flat, indices, tap_out, S, nblk, topk, win, Skv)
    torch.cuda.synchronize()
    mirror = tap_planes[0].clone()
    for p_i in range(1, nblk):
        mirror += tap_planes[p_i]
    for s in range(S):
        p_end = int(qpos_v[s].item()) + 1
        w_lo = max(p_end - win, 0)
        mirror[s, topk + (p_end - w_lo):] = float("-inf")  # never-written slots
        inc = indices[s].long()
        mirror[s, : topk][inc > qpos_v[s]] = float("-inf")  # sentinel/acausal
        dup = (inc >= w_lo) & (inc < p_end)
        mirror[s, topk + (inc[dup] - w_lo)] = float("-inf")
    assert bool((tap_out == mirror).all()), f"[{tag}] epilogue not bit-exact vs mirror"

    # Exact single-launch selection over the deduped union: slot top-k +
    # absolute-index mapping must reproduce the cascade_topk reference sets.
    top_v, top_slot = tap_out.topk(topk, dim=-1)
    w_lo_row = (qpos_v + 1 - win).clamp(min=0).view(-1, 1)
    abs_idx = torch.where(
        top_slot < topk,
        indices.long().gather(1, top_slot.clamp(max=topk - 1)),
        w_lo_row + (top_slot - topk),
    )
    abs_idx = torch.where(
        torch.isfinite(top_v), abs_idx, torch.full_like(abs_idx, -1)
    )
    agree_e = float(torch.tensor([
        len(set(abs_idx[i].tolist()) & set(ref_next[0, i].tolist())) / topk
        for i in range(S)
    ]).mean())
    assert agree_e >= 0.999, f"[{tag}] epilogue top-k disagrees: agreement={agree_e}"

    # Single-launch select kernel: the in-block bitonic top-k over the union
    # must reproduce the reference cascade sets exactly.
    next_idx = torch.full((S, topk), -1, dtype=torch.int32, device=dev)
    ext.launch_cascade_select(slab_flat, indices, next_idx, S, nblk, topk, win, Skv)
    torch.cuda.synchronize()
    agree_s = float(torch.tensor([
        len(set(next_idx[i].tolist()) & set(ref_next[0, i].tolist())) / topk
        for i in range(S)
    ]).mean())
    assert agree_s >= 0.999, f"[{tag}] select kernel disagrees: agreement={agree_s}"

    # Folded select (with_select=4, single launch): the last-arriver election
    # must reproduce the standalone select kernel's deterministic SET exactly
    # (order differs by atomic arrival), leave the attention output bitwise
    # mode-1 identical, and count exactly nblk*nsplit arrivals per row.
    next_fold = torch.full((S, topk), -1, dtype=torch.int32, device=dev)
    sel_counter = torch.zeros(S, dtype=torch.int32, device=dev)
    out.zero_()
    slab_flat.zero_()
    ext.launch_fused_sparse_splitk_selectfold(
        q, kv, indices, iqf8, ikf8, wp, sk, pa, pm, pl, slab_flat, hist,
        next_fold, sel_counter, out, S, Skv, H, Hidx, topk, nsplit,
        sm_scale, win)
    torch.cuda.synchronize()
    assert bool((out == out_mode1).all()), \
        f"[{tag}] fold mode attention not bitwise-identical to mode 1"
    assert bool((sel_counter == nblk * nsplit).all()), \
        f"[{tag}] fold election counted {sel_counter.tolist()} != {nblk * nsplit}"
    assert bool((next_fold.sort(-1).values == next_idx.sort(-1).values).all()), \
        f"[{tag}] folded select set differs from the select kernel"

    # Distilled tap weights (tapw entry): planes must hold the per-head
    # WEIGHTED ReLU sum (sum_h w_h ReLU(q_h . k)) and the attention output
    # must stay bitwise mode-1 (TapW only touches the selection planes).
    tap_w = torch.randn(H, device=dev).float().contiguous()
    out.zero_()
    slab_flat.zero_()
    ext.launch_fused_sparse_splitk_tapw(
        q, kv, indices, iqf8, ikf8, wp, sk, pa, pm, pl, slab_flat, hist,
        tap_w, out, S, Skv, H, Hidx, topk, nsplit, sm_scale, win)
    torch.cuda.synchronize()
    assert bool((out == out_mode1).all()), \
        f"[{tag}] tapw mode attention not bitwise-identical to mode 1"
    tapw_got = slab_flat[: nblk * S * ntap].view(nblk, S, ntap).sum(0)
    ref_tap_w = torch.zeros(S, ntap, device=dev)
    for s in range(S):
        p_end = int(qpos_v[s].item()) + 1
        w_lo = max(p_end - win, 0)
        inc = indices[s].long().clamp(max=Skv - 1)
        keys = torch.cat([inc, torch.arange(w_lo, w_lo + win, device=dev)])
        scores = torch.einsum("hd,kd->hk", qf[s], kv[0, keys, 0].float())
        valid = torch.cat([indices[s].long() <= qpos_v[s],
                           torch.arange(w_lo, w_lo + win, device=dev) < p_end])
        ref_tap_w[s] = (scores.relu() * tap_w.view(-1, 1)).sum(0) * valid.float()
    tw_corr = corr_diff(tapw_got, ref_tap_w)
    assert tw_corr < 1e-3, f"[{tag}] weighted tap planes disagree: 1-c={tw_corr}"

    # Pool/attend decoupling: the pool entry with attend_k == topk must be
    # bitwise-identical to the undecoupled tapw path (attention AND planes);
    # with attend_k < topk the planes must stay bitwise-identical (the tap
    # covers the whole pool regardless) while attention must match a plain
    # sparse run over only the leading attend_k columns.
    tapw_slab = slab_flat[: nblk * S * ntap].clone()
    out.zero_()
    slab_flat.zero_()
    ext.launch_fused_sparse_splitk_pool(
        q, kv, indices, iqf8, ikf8, wp, sk, pa, pm, pl, slab_flat, hist,
        tap_w, out, S, Skv, H, Hidx, topk, topk, nsplit, sm_scale, win)
    torch.cuda.synchronize()
    assert bool((out == out_mode1).all()), \
        f"[{tag}] pool entry (attend_k=topk) attention differs from mode 1"
    assert bool((slab_flat[: nblk * S * ntap] == tapw_slab).all()), \
        f"[{tag}] pool entry (attend_k=topk) planes differ from tapw path"

    # Off-tile attend_k (not a multiple of the BNA=32 tile) exercises the
    # straddle tile where the per-key gate, not the tile-level skip, must
    # separate attended from pool-only columns.
    attend_k = topk // 2 - 8
    out.zero_()
    slab_flat.zero_()
    ext.launch_fused_sparse_splitk_pool(
        q, kv, indices, iqf8, ikf8, wp, sk, pa, pm, pl, slab_flat, hist,
        tap_w, out, S, Skv, H, Hidx, topk, attend_k, nsplit, sm_scale, win)
    torch.cuda.synchronize()
    assert bool((slab_flat[: nblk * S * ntap] == tapw_slab).all()), \
        f"[{tag}] decoupled pool planes differ from the full-pool tap"
    ref_pool_attn = ref_sparse_mla(
        q, kv, indices[:, :attend_k].contiguous(), sm_scale, dev)
    p_corr = corr_diff(out.float(), ref_pool_attn)
    p_rel = float((out.float() - ref_pool_attn).abs().max()
                  / (ref_pool_attn.abs().max() + 1e-6))
    assert p_corr < 1e-3 and p_rel < 5e-2, \
        f"[{tag}] decoupled attention (attend_k={attend_k}) corr={p_corr} rel={p_rel}"
    # Bitwise gate on the attend boundary: excluding the pool tail via the
    # attend_k gate must produce attention bitwise-identical to excluding it
    # via Skv sentinels with NO gate (both are -inf lanes to the online
    # softmax and the split geometry is unchanged) — a one-key off-by-one in
    # the straddle gate cannot hide inside a correlation tolerance.
    out_gate = out.clone()
    idx_sent = indices.clone()
    idx_sent[:, attend_k:] = Skv
    out.zero_()
    slab_flat.zero_()
    ext.launch_fused_sparse_splitk_pool(
        q, kv, idx_sent, iqf8, ikf8, wp, sk, pa, pm, pl, slab_flat, hist,
        tap_w, out, S, Skv, H, Hidx, topk, topk, nsplit, sm_scale, win)
    torch.cuda.synchronize()
    assert bool((out == out_gate).all()), \
        f"[{tag}] attend_k gate is not equivalent to sentinel exclusion"
    # Re-run the gated configuration to restore the full-pool planes the
    # union-select gate below reads (the sentinel run zeroed the tail's).
    slab_flat.zero_()
    ext.launch_fused_sparse_splitk_pool(
        q, kv, indices, iqf8, ikf8, wp, sk, pa, pm, pl, slab_flat, hist,
        tap_w, out, S, Skv, H, Hidx, topk, attend_k, nsplit, sm_scale, win)
    torch.cuda.synchronize()

    # The decoupled union select over pool+window must agree with the
    # cascade_topk reference at pool width.
    tap_pool_got = slab_flat[: nblk * S * ntap].view(nblk, S, ntap).sum(0)
    got_pool_next = cascade_topk(cand_idx, tap_pool_got.unsqueeze(0), topk,
                                 query_positions=qpos_v)
    ref_pool_next = cascade_topk(cand_idx, ref_tap_w.unsqueeze(0), topk,
                                 query_positions=qpos_v)
    agree_p = float(torch.tensor([
        len(set(got_pool_next[0, i].tolist()) & set(ref_pool_next[0, i].tolist())) / topk
        for i in range(S)
    ]).mean())
    assert agree_p >= 0.999, f"[{tag}] pool next-indices disagree: agreement={agree_p}"

    logger.info("[%-14s nsplit=%2d] cascade: attn 1-c=%.2e | tap 1-c=%.2e | next-idx agree=%.4f "
                "| epilogue bit-exact, topk agree=%.4f | fold set-exact | tapw 1-c=%.2e "
                "| pool: planes bit-exact, attn 1-c=%.2e, next agree=%.4f",
                tag, nsplit, a3_corr, t_corr, agree3, agree_e, tw_corr, p_corr, agree_p)

    # with_select=5 (exp-tap / v4_probs): per-head raw-score planes + exp
    # epilogue.  H=16 additionally exercises the GLM pad geometry regardless
    # of the config's head count (dict.fromkeys dedupes while keeping order).
    for h_exp in dict.fromkeys([H, 16]):
        validate_exptap(ext, h_exp, dqk, Hidx, didx, S, Skv, nsplit, dev, tag)


def validate_exptap(ext, H, dqk, Hidx, didx, S, Skv, nsplit, dev, tag):
    """Gates for the exp-tap mode (with_select=5 + cascade_exp_epilogue).

    (a) attention output BITWISE identical to mode 3 — the attention path is
    untouched, only the plane writes differ; (b) exp-tap union logits — the
    epilogue plane columns PLUS host-scored sink columns using the exported
    per-head ExpStats, exactly the serving-path recipe — vs an eager fp32 v4
    reference (gather scores, per-head m/Z over the attend slice,
    exp-normalize, head-sum) at 2e-5 relative — an fp32 exp path, not
    bitwise; (c) tie-free selected-set agreement 1.0 over the full union
    (incumbents + window + sinks) vs the eager reference selection
    (continuous random scores make ties measure-zero).
    """
    from lagged_indexer_dsa.tap import cascade_topk

    topk = 2048
    win = 128
    nblk = H // 16
    ntap = topk + win
    sm_scale = dqk ** -0.5
    q = torch.randn(1, S, H, dqk, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(1, Skv, 1, dqk, dtype=torch.bfloat16, device=dev)
    iqf8, ikf8, wp, sk = make_fp8_indexer_inputs(S, Skv, Hidx, didx, dev)
    indices = make_indices(S, Skv, topk, dev)
    out = torch.zeros(1, S, H, 512, dtype=torch.bfloat16, device=dev)
    hist = torch.zeros(S, 256, dtype=torch.int32, device=dev)
    pa, pm, pl = alloc_part(S, H, nsplit, dev)
    slab3 = torch.zeros(
        max(S * Skv, nblk * S * ntap), dtype=torch.float32, device=dev)
    slab_h = torch.zeros(H * S * ntap, dtype=torch.float32, device=dev)

    # Mode-3 baseline for the bitwise attention gate.
    ext.launch_fused_sparse_splitk(q, kv, indices, iqf8, ikf8, wp, sk,
                                   pa, pm, pl, slab3, hist, out,
                                   S, Skv, H, Hidx, topk, nsplit, sm_scale,
                                   3, 1, win)
    torch.cuda.synchronize()
    out_mode3 = out.clone()

    out.zero_()
    ext.launch_fused_sparse_splitk_exptap(q, kv, indices, iqf8, ikf8, wp, sk,
                                          pa, pm, pl, slab_h, hist, out,
                                          S, Skv, H, Hidx, topk, nsplit,
                                          sm_scale, win)
    torch.cuda.synchronize()
    assert bool((out == out_mode3).all()), \
        f"[{tag} H={H}] exp-tap attention not bitwise-identical to mode 3"

    n_sink = 4
    tap_out = torch.zeros(S, ntap, dtype=torch.float32, device=dev)
    exp_stats = torch.zeros(S, 2, H, dtype=torch.float32, device=dev)
    ext.launch_cascade_exp_epilogue(slab_h, pm, pl, indices, tap_out,
                                    exp_stats, S, nblk, topk, win, Skv,
                                    nsplit)
    torch.cuda.synchronize()

    # Host-scored sink columns from the exported per-head stats — the exact
    # serving-path recipe: s2_sink in the kernel's log2-scaled domain
    # (score * sm_scale * log2e), then exp2(s2 - m2_g) / Z_g summed over
    # heads.  Commensurate with the plane columns by construction.
    qf = q[0].float()
    kvf = kv[0, :, 0, :].float()
    log2e = 1.4426950408889634
    s2_sink = torch.einsum("shd,kd->shk", qf, kvf[:n_sink]) \
        * (sm_scale * log2e)                          # [S, H, n_sink]
    m2g_t = exp_stats[:, 0, :].unsqueeze(-1)          # [S, H, 1]
    invz_t = exp_stats[:, 1, :].unsqueeze(-1)         # [S, H, 1]
    got_sink = torch.where(
        invz_t > 0, torch.exp2(s2_sink - m2g_t) * invz_t,
        torch.zeros_like(s2_sink),
    ).sum(1)                                          # [S, n_sink]

    # Eager fp32 v4 reference over the FULL candidate union (incumbents +
    # window + sinks): per-head (m, Z) over the causal ATTEND slice,
    # exp-normalize every candidate column, sum over heads.  Scores carry
    # sm_scale — the kernel's log2-domain s2 satisfies
    # exp2(s2 - m2) == exp(s*c - m).  One einsum per row so duplicated keys
    # yield bitwise-identical columns (cascade_topk's dedupe relies on it).
    nuni = ntap + n_sink
    qpos_v = torch.arange(S, device=dev) + (Skv - S)
    ref = torch.full((S, nuni), float("-inf"), device=dev)
    ref_raw = torch.zeros(S, nuni, device=dev)  # no dedupe, for cascade_topk
    win_idx = torch.zeros(S, win, dtype=torch.long, device=dev)
    sinks = torch.arange(n_sink, device=dev)
    for s in range(S):
        q_pos = int(qpos_v[s].item())
        p_end = q_pos + 1
        w_lo = max(p_end - win, 0)
        inc = indices[s].long()
        wkeys = torch.arange(w_lo, w_lo + win, device=dev)
        keys = torch.cat(
            [inc.clamp(max=Skv - 1), wkeys.clamp(max=Skv - 1), sinks])
        scores = torch.einsum("hd,kd->hk", qf[s], kvf[keys]) * sm_scale
        causal_inc = inc <= q_pos
        att = scores[:, :topk].masked_fill(
            ~causal_inc.view(1, -1), float("-inf"))
        m = att.amax(dim=-1, keepdim=True)             # [H, 1]
        z = (att - m).exp().sum(dim=-1, keepdim=True)  # [H, 1]
        vals = ((scores - m).exp() / z).sum(0)         # [nuni]
        valid = torch.cat([causal_inc, wkeys < p_end, sinks < p_end])
        ref_raw[s] = vals * valid.float()
        row = torch.where(valid, vals, torch.full_like(vals, float("-inf")))
        # Mirror the epilogue's dedupe: window copies of causal incumbents
        # are dropped (identical score by construction, lossless).
        dup = causal_inc & (inc >= w_lo)
        row[topk + (inc[dup] - w_lo)] = float("-inf")
        # Mirror sink_duplicate_mask: a sink already present as incumbent or
        # covered by the window enters the union exactly once.
        sink_dup = (inc.view(-1, 1) == sinks.view(1, -1)).any(0) | (sinks >= w_lo)
        row[ntap:][sink_dup] = float("-inf")
        ref[s] = row
        win_idx[s] = wkeys

    # Full union on the got side: epilogue plane columns + host sink columns
    # with the same duplicate mask the serving path applies.
    w_lo_row = (qpos_v + 1 - win).clamp(min=0).view(-1, 1)
    sink_dup_all = (
        (indices.long().unsqueeze(-1) == sinks.view(1, 1, -1)).any(1)
        | (sinks.view(1, -1) >= w_lo_row)
    )
    union_got = torch.cat(
        [tap_out, got_sink.masked_fill(sink_dup_all, float("-inf"))], dim=1)

    finite_got = torch.isfinite(union_got)
    finite_ref = torch.isfinite(ref)
    assert bool((finite_got == finite_ref).all()), \
        f"[{tag} H={H}] exp-tap -inf placement differs from the reference"
    e_rel = float((union_got[finite_got] - ref[finite_ref]).abs().max()
                  / ref[finite_ref].abs().max().clamp(min=1e-20))
    assert e_rel <= 2e-5, \
        f"[{tag} H={H}] exp-tap logits rel={e_rel:.3e} > 2e-5"

    # Tie-free selected-set agreement over the full union vs the eager
    # reference selection.
    cand_idx = torch.cat(
        [indices.long(), win_idx, sinks.view(1, -1).expand(S, -1)],
        dim=1).unsqueeze(0)
    ref_next = cascade_topk(cand_idx, ref_raw.unsqueeze(0), topk,
                            query_positions=qpos_v)
    top_v, top_slot = union_got.topk(topk, dim=-1)
    in_inc = top_slot < topk
    in_win = (top_slot >= topk) & (top_slot < ntap)
    abs_idx = torch.where(
        in_inc,
        indices.long().gather(1, top_slot.clamp(max=topk - 1)),
        torch.where(in_win, w_lo_row + (top_slot - topk), top_slot - ntap),
    )
    abs_idx = torch.where(
        torch.isfinite(top_v), abs_idx, torch.full_like(abs_idx, -1))
    # Exact per-row set equality over VALID keys.  The old |∩|/topk metric
    # could never reach 1.0 when prefix < topk (both sides select at most
    # `prefix` valid keys + the shared -1 sentinel: job 20735's shortpfx
    # "0.5005" was 1025/2048 with every valid key agreeing) — and set
    # equality is strictly stronger at long prefixes, where the old 1.0
    # bar is equivalent.
    for i in range(S):
        got_set = {x for x in abs_idx[i].tolist() if x >= 0}
        ref_set = {x for x in ref_next[0, i].tolist() if x >= 0}
        assert got_set == ref_set, (
            f"[{tag} H={H}] exp-tap select set mismatch row {i}: "
            f"|got|={len(got_set)} |ref|={len(ref_set)} "
            f"symdiff={len(got_set ^ ref_set)}"
        )
    agree = 1.0

    logger.info("[%-14s nsplit=%2d] exp-tap H=%3d: attn bitwise mode-3 | "
                "union (incl %d sinks) rel=%.2e | select agree=%.4f",
                tag, nsplit, H, n_sink, e_rel, agree)


def validate_batched(ext, H, dqk, Hidx, didx, B, Skv, nsplit, dev, tag):
    """Batched-decode gates: kv_stride launch == B single-row launches, BITWISE.

    Builds B independent sequences in one [B, Skv, dqk] buffer and drives the
    batched entry (row b reads KV + b*kv_stride, q_pos = Skv-1 on every row),
    then replays each row through the certified legacy single-row entries.
    Per-row arithmetic is identical by construction, so every comparison is
    exact equality, not a tolerance: attention output, tap planes, epilogue
    tap logits (cascade mode 3, weighted + windowed), attention-only mode 0,
    and pool/attend decoupling (attend_k < topk).
    """
    topk = 2048
    win = 128
    nblk = H // 16
    ntap = topk + win
    sm_scale = dqk ** -0.5
    kv_stride = Skv * dqk

    q = torch.randn(1, B, H, dqk, dtype=torch.bfloat16, device=dev)
    kvb = torch.randn(B, Skv, dqk, dtype=torch.bfloat16, device=dev)
    indices = torch.stack([
        torch.randperm(Skv, device=dev)[:topk].to(torch.int32) for _ in range(B)
    ])
    tapw = (torch.rand(H, dtype=torch.float32, device=dev) + 0.5).contiguous()
    # fp8 indexer operands are never dereferenced in modes 0/3; the entry
    # points still take the tensors, so build row-sized ones once.
    iqf8, ikf8, wp, sk = make_fp8_indexer_inputs(B, Skv, Hidx, didx, dev)

    out_b = torch.zeros(1, B, H, 512, dtype=torch.bfloat16, device=dev)
    slab_b = torch.zeros(nblk * B * ntap, dtype=torch.float32, device=dev)
    tap_b = torch.zeros(B, ntap, dtype=torch.float32, device=dev)
    pa, pm, pl = alloc_part(B, H, nsplit, dev)
    hist = torch.zeros(B, 256, dtype=torch.int32, device=dev)

    def _run_batch(with_select: int, attend_k: int) -> None:
        out_b.zero_(); slab_b.zero_(); tap_b.zero_()
        ext.launch_fused_sparse_splitk_batch(
            q, kvb, indices, iqf8, ikf8, wp, sk, pa, pm, pl, slab_b, hist,
            tapw, out_b, B, Skv, H, Hidx, topk, attend_k, nsplit, sm_scale,
            with_select, win if with_select == 3 else 0, kv_stride)
        if with_select == 3:
            ext.launch_cascade_epilogue_batch(
                slab_b, indices, tap_b, B, nblk, topk, win, Skv)
        torch.cuda.synchronize()

    pa1, pm1, pl1 = alloc_part(1, H, nsplit, dev)
    hist1 = torch.zeros(1, 256, dtype=torch.int32, device=dev)

    def _run_row(b: int, with_select: int, attend_k: int):
        """Legacy single-row launch on row b's KV slice; returns (out, planes, tap)."""
        out_1 = torch.zeros(1, 1, H, 512, dtype=torch.bfloat16, device=dev)
        slab_1 = torch.zeros(nblk * ntap, dtype=torch.float32, device=dev)
        tap_1 = torch.zeros(1, ntap, dtype=torch.float32, device=dev)
        qr = q[:, b : b + 1].contiguous()
        kvr = kvb[b].view(1, Skv, 1, dqk)
        idxr = indices[b : b + 1]
        iq1, wp1 = iqf8[b : b + 1].contiguous(), wp[b : b + 1].contiguous()
        if with_select == 0:
            ext.launch_fused_sparse_splitk(
                qr, kvr, idxr, iq1, ikf8, wp1, sk, pa1, pm1, pl1, slab_1,
                hist1, out_1, 1, Skv, H, Hidx, topk, nsplit, sm_scale, 0, 1, 0)
        elif attend_k == topk:
            ext.launch_fused_sparse_splitk_tapw(
                qr, kvr, idxr, iq1, ikf8, wp1, sk, pa1, pm1, pl1, slab_1,
                hist1, tapw, out_1, 1, Skv, H, Hidx, topk, nsplit, sm_scale, win)
        else:
            ext.launch_fused_sparse_splitk_pool(
                qr, kvr, idxr, iq1, ikf8, wp1, sk, pa1, pm1, pl1, slab_1,
                hist1, tapw, out_1, 1, Skv, H, Hidx, topk, attend_k, nsplit,
                sm_scale, win)
        if with_select == 3:
            ext.launch_cascade_epilogue(
                slab_1, idxr, tap_1, 1, nblk, topk, win, Skv)
        torch.cuda.synchronize()
        return out_1, slab_1.view(nblk, 1, ntap), tap_1

    for with_select, attend_k, mode_tag in (
        (3, topk, "cascade"), (0, topk, "attn-only"), (3, topk // 2, "pool"),
    ):
        _run_batch(with_select, attend_k)
        out_snap = out_b.clone()
        planes_snap = slab_b.clone().view(nblk, B, ntap)
        tap_snap = tap_b.clone()
        for b in range(B):
            out_1, planes_1, tap_1 = _run_row(b, with_select, attend_k)
            assert bool((out_snap[:, b] == out_1[:, 0]).all()), (
                f"[{tag}] {mode_tag}: batched attention row {b} not bitwise-"
                "equal to its single-row launch")
            if with_select == 3:
                assert bool((planes_snap[:, b] == planes_1[:, 0]).all()), (
                    f"[{tag}] {mode_tag}: batched tap planes row {b} not "
                    "bitwise-equal to its single-row launch")
                assert bool((tap_snap[b] == tap_1[0]).all()), (
                    f"[{tag}] {mode_tag}: batched epilogue row {b} not "
                    "bitwise-equal to its single-row launch")
    # R2b: batched fused-indexer mode (with_select=1) via the batch_idx
    # entry, against per-row legacy launches — bitwise on BOTH outputs
    # (attention Out and the full-prefix SelSlab logits).  Per-row index
    # keys/scales: [B, Skv, didx] / [B, Skv] with row strides Skv*didx / Skv.
    ikb = torch.stack([
        make_fp8_indexer_inputs(1, Skv, Hidx, didx, dev)[1] for _ in range(B)
    ])  # [B, Skv, didx] fp8 — independent keys per row
    skb = torch.rand(B, Skv, dtype=torch.float32, device=dev) + 0.5
    slab_idx = torch.zeros(B * Skv, dtype=torch.float32, device=dev)
    out_b.zero_()
    ext.launch_fused_sparse_splitk_batch_idx(
        q, kvb, indices, iqf8, ikb, wp, skb, pa, pm, pl, slab_idx, hist,
        out_b, B, Skv, H, Hidx, topk, nsplit, sm_scale,
        1, 1, 0, kv_stride, Skv * didx, Skv)
    torch.cuda.synchronize()
    out_snap = out_b.clone()
    slab_snap = slab_idx.clone().view(B, Skv)
    for b in range(B):
        out_1 = torch.zeros(1, 1, H, 512, dtype=torch.bfloat16, device=dev)
        slab_1 = torch.zeros(Skv, dtype=torch.float32, device=dev)
        qr = q[:, b : b + 1].contiguous()
        kvr = kvb[b].view(1, Skv, 1, dqk)
        ext.launch_fused_sparse_splitk(
            qr, kvr, indices[b : b + 1], iqf8[b : b + 1].contiguous(),
            ikb[b], wp[b : b + 1].contiguous(), skb[b].contiguous(),
            pa1, pm1, pl1, slab_1, hist1, out_1,
            1, Skv, H, Hidx, topk, nsplit, sm_scale, 1, 1, 0)
        torch.cuda.synchronize()
        assert bool((out_snap[:, b] == out_1[:, 0]).all()), (
            f"[{tag}] idx-batched: attention row {b} not bitwise-equal to "
            "its single-row launch")
        assert bool((slab_snap[b] == slab_1).all()), (
            f"[{tag}] idx-batched: SelSlab logits row {b} not bitwise-equal "
            "to its single-row launch")

    logger.info("[%-14s nsplit=%2d] batched B=%d: cascade/attn-only/pool/"
                "idx-select(1) all bitwise-equal to single-row launches",
                tag, nsplit, B)


def validate_exptap_batched(ext, H, dqk, Hidx, didx, B, Skv, nsplit, dev, tag):
    """Batched exp-tap gates: kv_stride mode-5 launch == B single-row launches.

    Same certification pattern as ``validate_batched``, for the exp-tap mode
    (with_select=5 + cascade_exp_epilogue): B independent sequences through
    the batched entry vs B replays through the certified single-row exptap
    entry + legacy epilogue.  Per-row arithmetic is identical by construction
    (the mode-5 plane writes index by row exactly like mode 3, and the
    epilogue's per-head (m, Z) reduction reads the same per-row PartM/PartL
    blocks), so every comparison is exact equality: attention output,
    per-head raw-score planes, epilogue union logits, and the exported
    per-head ExpStats the host sink-scoring consumes.  Row 0 carries Skv
    sentinels in its incumbent tail so the -inf placement path is exercised.
    """
    topk = 2048
    win = 128
    nblk = H // 16
    ntap = topk + win
    sm_scale = dqk ** -0.5
    kv_stride = Skv * dqk

    q = torch.randn(1, B, H, dqk, dtype=torch.bfloat16, device=dev)
    kvb = torch.randn(B, Skv, dqk, dtype=torch.bfloat16, device=dev)
    indices = torch.stack([
        torch.randperm(Skv, device=dev)[:topk].to(torch.int32) for _ in range(B)
    ])
    indices[0, -5:] = Skv  # sentinel slots: epilogue must -inf them
    iqf8, ikf8, wp, sk = make_fp8_indexer_inputs(B, Skv, Hidx, didx, dev)

    out_b = torch.zeros(1, B, H, 512, dtype=torch.bfloat16, device=dev)
    slab_h_b = torch.zeros(H * B * ntap, dtype=torch.float32, device=dev)
    tap_b = torch.zeros(B, ntap, dtype=torch.float32, device=dev)
    stats_b = torch.zeros(B, 2, H, dtype=torch.float32, device=dev)
    pa, pm, pl = alloc_part(B, H, nsplit, dev)
    hist = torch.zeros(B, 256, dtype=torch.int32, device=dev)

    ext.launch_fused_sparse_splitk_exptap_batch(
        q, kvb, indices, iqf8, ikf8, wp, sk, pa, pm, pl, slab_h_b, hist,
        out_b, B, Skv, H, Hidx, topk, nsplit, sm_scale, win, kv_stride)
    ext.launch_cascade_exp_epilogue_batch(
        slab_h_b, pm, pl, indices, tap_b, stats_b, B, nblk, topk, win, Skv,
        nsplit)
    torch.cuda.synchronize()
    planes_b = slab_h_b.view(H, B, ntap)

    pa1, pm1, pl1 = alloc_part(1, H, nsplit, dev)
    hist1 = torch.zeros(1, 256, dtype=torch.int32, device=dev)
    for b in range(B):
        out_1 = torch.zeros(1, 1, H, 512, dtype=torch.bfloat16, device=dev)
        slab_h_1 = torch.zeros(H * ntap, dtype=torch.float32, device=dev)
        tap_1 = torch.zeros(1, ntap, dtype=torch.float32, device=dev)
        stats_1 = torch.zeros(1, 2, H, dtype=torch.float32, device=dev)
        qr = q[:, b : b + 1].contiguous()
        kvr = kvb[b].view(1, Skv, 1, dqk)
        idxr = indices[b : b + 1]
        iq1, wp1 = iqf8[b : b + 1].contiguous(), wp[b : b + 1].contiguous()
        ext.launch_fused_sparse_splitk_exptap(
            qr, kvr, idxr, iq1, ikf8, wp1, sk, pa1, pm1, pl1, slab_h_1,
            hist1, out_1, 1, Skv, H, Hidx, topk, nsplit, sm_scale, win)
        ext.launch_cascade_exp_epilogue(
            slab_h_1, pm1, pl1, idxr, tap_1, stats_1, 1, nblk, topk, win,
            Skv, nsplit)
        torch.cuda.synchronize()
        assert bool((out_b[:, b] == out_1[:, 0]).all()), (
            f"[{tag}] exp-tap: batched attention row {b} not bitwise-equal "
            "to its single-row launch")
        assert bool((planes_b[:, b] == slab_h_1.view(H, 1, ntap)[:, 0]).all()), (
            f"[{tag}] exp-tap: batched per-head planes row {b} not "
            "bitwise-equal to its single-row launch")
        assert bool((tap_b[b] == tap_1[0]).all()), (
            f"[{tag}] exp-tap: batched epilogue logits row {b} not "
            "bitwise-equal to its single-row launch")
        assert bool((stats_b[b] == stats_1[0]).all()), (
            f"[{tag}] exp-tap: batched ExpStats row {b} not bitwise-equal "
            "to its single-row launch")
    logger.info("[%-14s nsplit=%2d] batched exp-tap H=%3d B=%d: attention/"
                "planes/epilogue/stats all bitwise-equal to single-row "
                "launches", tag, nsplit, H, B)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skv", type=int, nargs="+", default=[4096, 8192, 16384, 32768, 65536, 131072])
    p.add_argument("--batch", type=int, nargs="+", default=[1],
                   help="query-row counts S to sweep (decode-batch / occupancy proxy; "
                        "the grid is S*nblk*nsplit blocks, so larger S saturates SMs)")
    p.add_argument("--nsplit", type=int, nargs="+", default=[16, 32])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default="deepseek", choices=["deepseek", "glm5", "glm5.1"])
    p.add_argument("--out", type=str, default=None,
                   help="optional path to write the benchmark sweep as JSON")
    p.add_argument("--cascade-budget", action="store_true",
                   help="Also measure the tap-cascade decode budget: attention-only + "
                        "admission indexer over W keys + (topk+W)-candidate top-k, vs "
                        "today's fused + from-hist selector pipeline.")
    p.add_argument("--admission-window", type=int, nargs="+", default=[128, 1024],
                   help="Cascade admission-window sizes W to measure")
    p.add_argument("--profile-idx", type=int, default=0,
                   help="if >0: run only this many indexer-only launches at skv[0] "
                        "(no validate/sweep) so Nsight Compute can profile a clean kernel")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(args.seed)

    cfg = named_config(args.config)
    H, dqk = cfg.num_attention_heads, cfg.qk_head_dim
    Hidx, didx = cfg.index_n_heads, cfg.index_head_dim
    topk = cfg.index_topk
    dev = torch.device("cuda")
    ext = build()
    logger.info("device=%s | H=%d dqk=%d Hidx=%d didx=%d topk=%d nsplit=%s",
                torch.cuda.get_device_name(0), H, dqk, Hidx, didx, topk, args.nsplit)

    if args.profile_idx > 0:
        Skv = args.skv[0]
        S = 1
        q = torch.randn(1, S, H, dqk, dtype=torch.bfloat16, device=dev)
        kv = torch.randn(1, Skv, 1, dqk, dtype=torch.bfloat16, device=dev)
        iqf8, ikf8, wp, sk = make_fp8_indexer_inputs(S, Skv, Hidx, didx, dev)
        indices = make_indices(S, Skv, topk, dev)
        out = torch.zeros(1, S, H, 512, dtype=torch.bfloat16, device=dev)
        slab = torch.zeros(S, Skv, dtype=torch.float32, device=dev)
        pa, pm, pl = alloc_part(S, H, args.nsplit[0], dev)
        hist = torch.zeros(S, 256, dtype=torch.int32, device=dev)
        sm_scale = dqk ** -0.5
        for _ in range(args.profile_idx):
            ext.launch_fused_sparse_splitk(q, kv, indices, iqf8, ikf8, wp, sk, pa, pm, pl,
                                           slab, hist, out, S, Skv, H, Hidx, topk, args.nsplit[0],
                                           sm_scale, 1, 0, 0)   # with_select=1, attn_on=0
        torch.cuda.synchronize()
        logger.info("profile-idx: ran %d idx-only launches at Skv=%d", args.profile_idx, Skv)
        return

    for ns in args.nsplit:
        for S, Skv, tag in [(1, 8192, "decode 8k"), (1, 32768, "decode 32k"), (4, 8192, "decode x4")]:
            validate(ext, H, dqk, Hidx, didx, S, Skv, ns, dev, tag)
        for B, Skv, tag in [(4, 8192, "batch x4 8k"), (8, 16384, "batch x8 16k")]:
            validate_batched(ext, H, dqk, Hidx, didx, B, Skv, ns, dev, tag)
            # Exp-tap batched rows at the config's head count AND the GLM
            # rank-pad geometry (H=16), matching validate()'s exp-tap sweep.
            for h_exp in dict.fromkeys([H, 16]):
                validate_exptap_batched(
                    ext, h_exp, dqk, Hidx, didx, B, Skv, ns, dev, tag)
    logger.info("VALIDATION OK")

    # Keyed by batch size: the admission indexer and candidate top-k scale
    # with the number of decode rows, so measure them per batch.
    cascade_parts_by_batch: dict[int, dict[int, dict[str, float]]] = {}
    if args.cascade_budget:
        for bs in args.batch:
            cascade_parts_by_batch[bs] = measure_cascade_parts(
                ext, dev, args.admission_window, args.nsplit[0],
                H, dqk, Hidx, didx, topk, S=bs,
            )

    sweep_rows: list[dict] = []

    def bench(Skv, nsplit, S=1, iters=50):
        q = torch.randn(1, S, H, dqk, dtype=torch.bfloat16, device=dev)
        kv = torch.randn(1, Skv, 1, dqk, dtype=torch.bfloat16, device=dev)
        iqf8, ikf8, wp, sk = make_fp8_indexer_inputs(S, Skv, Hidx, didx, dev)
        indices = make_indices(S, Skv, topk, dev)
        out = torch.zeros(1, S, H, 512, dtype=torch.bfloat16, device=dev)
        # Flat slab sized for both layouts: legacy [S, Skv] logits and the
        # cascade's per-head-group tap planes [nblk, S, topk + max(W)].
        nblk_l = H // 16
        max_w = max(args.admission_window) if args.cascade_budget else 0
        slab_flat = torch.zeros(
            max(S * Skv, nblk_l * S * (topk + max_w)),
            dtype=torch.float32, device=dev)
        slab = slab_flat[: S * Skv].view(S, Skv)
        pa, pm, pl = alloc_part(S, H, nsplit, dev)
        sm_scale = dqk ** -0.5

        from lagged_indexer_dsa.tilelang_dsa.topk_selector import tl_topk

        hist = torch.zeros(S, 256, dtype=torch.int32, device=dev)
        starts = torch.zeros(S, dtype=torch.int32, device=dev)
        ends = (torch.arange(S, device=dev) + (Skv - S) + 1).to(torch.int32)

        def kernel(with_select, attn_on, win=0):
            ext.launch_fused_sparse_splitk(q, kv, indices, iqf8, ikf8, wp, sk, pa, pm, pl, slab_flat, hist, out,
                                           S, Skv, H, Hidx, topk, nsplit, sm_scale, with_select, attn_on, win)

        def time_fn(fn):
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            st, en = torch.cuda.Event(True), torch.cuda.Event(True)
            st.record()
            for _ in range(iters):
                fn()
            en.record()
            torch.cuda.synchronize()
            return st.elapsed_time(en) / iters

        def run(with_select, attn_on):
            return time_fn(lambda: kernel(with_select, attn_on))

        attn = run(0, 1)
        idx = run(1, 0)
        fused = run(1, 1)
        # matching kernel baseline for the from-hist tail decomposition (the
        # with_select=2 kernel also zeroes+builds the histogram)
        fused2 = run(2, 1)

        # strict=False for timing only: the fp8-quantized logit distribution
        # is bucket-dense enough that the threshold bucket overflows the
        # 8192-slot buffer at 131k (~400 candidates over) — the exactness
        # gates live in validate(), which stays strict.
        def pipe_legacy():
            kernel(1, 1)
            tl_topk(slab, starts, ends, topk, strict=False)

        def pipe_fused():
            kernel(2, 1)
            tl_topk(slab, starts, ends, topk, hist=hist, strict=False)

        pipel = time_fn(pipe_legacy)
        pipef = time_fn(pipe_fused)

        model_max = max(attn, idx)
        serial = attn + idx
        resid = 100 * (fused - model_max) / model_max
        saved = 100 * (serial - fused) / serial
        logger.info("B=%-3d Skv=%-7d attn=%.3f idx=%.3f | serial=%.3f fused=%.3f max=%.3f "
                    "| fused-vs-max %+.1f%% | saved-vs-serial %+.1f%%",
                    S, Skv, attn, idx, serial, fused, model_max, resid, saved)
        logger.info("B=%-3d Skv=%-7d pipeline fused+topk: legacy=%.3f from-hist=%.3f ms "
                    "(topk tail %.3f -> %.3f, matched kernel baselines)",
                    S, Skv, pipel, pipef, pipel - fused, pipef - fused2)
        cascade_cols: dict[str, float] = {}
        for W, part in cascade_parts_by_batch.get(S, {}).items():
            # Composed lower bound: attention-only + separate admission
            # indexer launch + candidate top-k (launch overhead included).
            tk = min(part["topk_tl_ms"], part["topk_torch_ms"])
            casc = attn + part["admission_ms"] + tk
            cascade_cols[f"cascade_w{W}_ms"] = casc
            cascade_cols[f"cascade_w{W}_speedup_vs_pipeline"] = pipef / casc
            logger.info(
                "B=%-3d Skv=%-7d CASCADE W=%-5d attn=%.3f + adm=%.3f + topk=%.3f "
                "= %.3f ms | today=%.3f ms | speedup %.2fx",
                S, Skv, W, attn, part["admission_ms"], tk, casc, pipef, pipef / casc)
        if args.cascade_budget:
            # Fused cascade (with_select=3): tap over incumbent + admission
            # window in the SAME launch; host sums the head-group planes and
            # takes the deduped union top-k — the complete tap→indices path.
            from lagged_indexer_dsa.tap import cascade_topk

            qpos_v = torch.arange(S, device=dev) + (Skv - S)
            for W in args.admission_window:
                ntap = topk + W
                planes = slab_flat[: nblk_l * S * ntap].view(nblk_l, S, ntap)
                win_idx = (qpos_v.view(-1, 1) + 1 - W
                           + torch.arange(W, device=dev).view(1, -1)).long()
                cand_idx = torch.cat([indices.long(), win_idx], dim=1).unsqueeze(0)

                def casc_pipe():
                    kernel(3, 1, W)
                    tap = planes.sum(0)
                    cascade_topk(cand_idx, tap.unsqueeze(0), topk,
                                 query_positions=qpos_v)

                def casc_fast_tail():
                    # dup-tolerant tail: single-launch top-k over the union
                    # (duplicate keys carry identical scores; effective k
                    # shrinks by |incumbent ∩ window| — bound reported).
                    kernel(3, 1, W)
                    tap = planes.sum(0)
                    torch.topk(tap, topk, dim=-1)

                casck = time_fn(lambda: kernel(3, 1, W))  # kernel only
                cascf = time_fn(casc_pipe)
                cascft = time_fn(casc_fast_tail)

                # Headline exact path: main kernel + epilogue (plane-sum +
                # dedupe) + single torch.topk + absolute-index map.
                ntap_w = topk + W
                tap_out = torch.zeros(S, ntap_w, dtype=torch.float32, device=dev)
                top_v = torch.empty(S, topk, dtype=torch.float32, device=dev)
                top_slot = torch.empty(S, topk, dtype=torch.int64, device=dev)
                w_lo_row = (qpos_v + 1 - W).clamp(min=0).view(-1, 1)
                idx_long = indices.long()

                def casc_exact():
                    kernel(3, 1, W)
                    ext.launch_cascade_epilogue(slab_flat, indices, tap_out,
                                                S, nblk_l, topk, W, Skv)
                    torch.topk(tap_out, topk, dim=-1, out=(top_v, top_slot))
                    abs_i = torch.where(
                        top_slot < topk,
                        idx_long.gather(1, top_slot.clamp(max=topk - 1)),
                        w_lo_row + (top_slot - topk))
                    return torch.where(torch.isfinite(top_v), abs_i,
                                       torch.full_like(abs_i, -1))

                def _capture(fn, check, canon=None):
                    # canon: canonicalize outputs before the replay-vs-eager
                    # equality check (the histogram-select kernel emits a
                    # set-exact but atomic-arrival-ordered index list, so its
                    # gate compares sorted sets; deterministic paths compare
                    # bitwise).
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(3):
                            fn()
                    torch.cuda.current_stream().wait_stream(side)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        gout = fn()
                    eout = fn()
                    torch.cuda.synchronize()
                    esnap = check().clone() if gout is None else eout.clone()
                    g.replay()
                    torch.cuda.synchronize()
                    rsnap = check().clone() if gout is None else gout.clone()
                    if canon is not None:
                        esnap, rsnap = canon(esnap), canon(rsnap)
                    assert bool((rsnap == esnap).all()), \
                        "graph replay differs from eager execution"
                    return g

                casce = time_fn(casc_exact)
                graph = _capture(casc_exact, lambda: top_slot)
                cascg = time_fn(graph.replay)

                # Two-launch select chain: mode-3 kernel + the standalone
                # three-level radix select (deterministic tie-break).
                next_idx = torch.full((S, topk), -1, dtype=torch.int32, device=dev)

                def casc_chain():
                    kernel(3, 1, W)
                    ext.launch_cascade_select(slab_flat, indices, next_idx,
                                              S, nblk_l, topk, W, Skv)

                chain_graph = _capture(casc_chain, lambda: next_idx,
                                       canon=lambda t: t.sort(dim=-1).values)
                cascs = time_fn(chain_graph.replay)

                # Folded chain: ONE launch — fused attention + tap planes +
                # last-arriver radix select — plus the combine reduction. The
                # gap to `cascs` is the select kernel's launch boundary.
                next_idx_fold = torch.full((S, topk), -1, dtype=torch.int32,
                                           device=dev)
                sel_counter = torch.zeros(S, dtype=torch.int32, device=dev)

                def casc_fold():
                    ext.launch_fused_sparse_splitk_selectfold(
                        q, kv, indices, iqf8, ikf8, wp, sk, pa, pm, pl,
                        slab_flat, hist, next_idx_fold, sel_counter, out,
                        S, Skv, H, Hidx, topk, nsplit, sm_scale, W)

                # Parity gate: the folded select must reproduce the two-launch
                # chain's deterministic set (order differs by atomic arrival).
                casc_chain()
                casc_fold()
                torch.cuda.synchronize()
                assert bool((next_idx_fold.sort(-1).values
                             == next_idx.sort(-1).values).all()), \
                    "folded select disagrees with the two-launch chain"
                fold_graph = _capture(casc_fold, lambda: next_idx_fold,
                                      canon=lambda t: t.sort(dim=-1).values)
                cascfold = time_fn(fold_graph.replay)

                # Fairness: graph-capture the baseline pipeline too when the
                # tilelang selector allows it; otherwise say so loudly.
                pipeg = None
                try:
                    bside = torch.cuda.Stream()
                    bside.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(bside):
                        for _ in range(3):
                            pipe_fused()
                    torch.cuda.current_stream().wait_stream(bside)
                    torch.cuda.synchronize()
                    bgraph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(bgraph):
                        pipe_fused()
                    pipeg = time_fn(bgraph.replay)
                except RuntimeError as exc:
                    logger.warning(
                        "baseline pipeline not graph-capturable (%s); "
                        "graphed-cascade compares against the eager baseline",
                        exc)
                base_best = min(pipef, pipeg) if pipeg is not None else pipef

                cascade_cols[f"fused_cascade_w{W}_kernel_ms"] = casck
                cascade_cols[f"fused_cascade_w{W}_ms"] = cascf
                cascade_cols[f"fused_cascade_w{W}_fast_tail_ms"] = cascft
                cascade_cols[f"fused_cascade_w{W}_exact_ms"] = casce
                cascade_cols[f"fused_cascade_w{W}_graph_ms"] = cascg
                cascade_cols[f"fused_cascade_w{W}_sortchain_graph_ms"] = cascs
                cascade_cols[f"fused_cascade_w{W}_foldchain_graph_ms"] = cascfold
                cascade_cols[f"pipeline_graph_w{W}_ms"] = pipeg
                cascade_cols[f"fused_cascade_w{W}_speedup_vs_pipeline"] = pipef / casce
                cascade_cols[f"fused_cascade_w{W}_speedup_graphed"] = base_best / cascg
                cascade_cols[f"fused_cascade_w{W}_speedup_fold_graphed"] = base_best / cascfold
                logger.info(
                    "B=%-3d Skv=%-7d FUSED-CASCADE W=%-5d kernel=%.3f (attn-only=%.3f, "
                    "sel-cost %+.1f%%) | exact %.3f | GRAPHED %.3f | sortchain %.3f "
                    "| FOLD %.3f ms | today=%.3f (graphed %s) | speedup eager %.2fx "
                    "graphed %.2fx fold %.2fx",
                    S, Skv, W, casck, attn, 100 * (casck - attn) / attn,
                    casce, cascg, cascs, cascfold, pipef,
                    f"{pipeg:.3f}" if pipeg is not None else "n/a",
                    pipef / casce, base_best / cascg, base_best / cascfold)
        sweep_rows.append({
            **cascade_cols,
            "context_len": Skv,
            "batch": S,
            "nsplit": nsplit,
            "attn_only_ms": attn,
            "indexer_only_ms": idx,
            "serial_ms": serial,
            "fused_ms": fused,
            "ideal_overlap_ms": model_max,
            "fused_vs_ideal_pct": resid,
            "saved_vs_serial_pct": saved,
            "pipeline_topk_legacy_ms": pipel,
            "pipeline_topk_from_hist_ms": pipef,
        })

    for ns in args.nsplit:
        for S in args.batch:
            for skv in args.skv:
                bench(skv, ns, S)

    if args.out:
        import json

        payload = {
            "device": torch.cuda.get_device_name(0),
            "config": args.config,
            "num_attention_heads": H,
            "qk_head_dim": dqk,
            "index_n_heads": Hidx,
            "index_head_dim": didx,
            "index_topk": topk,
            "nsplit": args.nsplit if isinstance(args.nsplit, list) else [args.nsplit],
            "seed": args.seed,
            "admission_windows": args.admission_window if args.cascade_budget else None,
            "cascade_parts_by_batch": cascade_parts_by_batch or None,
            "rows": sweep_rows,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
