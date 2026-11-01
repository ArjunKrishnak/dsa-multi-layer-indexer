import torch
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}


def convert_to_uint16(x):
    hval = T.cast(x, T.float16)
    bits_uint = T.reinterpret(hval, T.uint16)
    bits_uint = T.if_then_else(x < 0, ~bits_uint & (0xFFFF), bits_uint | (0x8000))
    return bits_uint >> 8


def torch_bucket16(x):
    """torch mirror of convert_to_uint16 (and the CUDA benches' sel_bucket16):
    the radix bucket a value lands in, for building/validating external
    histograms passed to tl_topk(hist=...)."""
    bits = x.half().view(torch.int16).to(torch.int32) & 0xFFFF
    bits = torch.where(x < 0, ~bits & 0xFFFF, bits | 0x8000)
    return bits >> 8


def convert_to_uint32(x):
    bits_uint = T.reinterpret(x, T.uint32)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.cast((0xFFFFFFFF), T.uint32),
        bits_uint | T.cast((0x80000000), T.uint32),
    )
    return bits_uint


@tilelang.jit(pass_configs=pass_configs)
def tl_topk_impl(input, index, starts, ends, overflow, in_dtype=T.float32, out_dtype=T.int32):
    topk = T.const("topk")
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    RADIX = 1 << 8
    BLOCK_SIZE = 1024
    # Candidate buffer for the threshold bucket (all elements sharing the k-th
    # value's top-8 fp16 bits). The original 4096 was an UNCHECKED assumption:
    # at ctx=262144 with unit-normal logits the bucket holds ~4300 entries and
    # the atomic collect wrote out of shared-memory bounds (illegal memory
    # access). 8192 + bounds checks below; overflow[bx] counts dropped
    # candidates so callers can fail fast instead of silently degrading.
    SMEM_INPUT_SIZE = 8192

    input: T.Tensor[(batch, seq_len), in_dtype]
    index: T.Tensor[(batch, topk), out_dtype]
    starts: T.Tensor[(batch), out_dtype]
    ends: T.Tensor[(batch), out_dtype]
    overflow: T.Tensor[(batch), out_dtype]

    with T.Kernel(batch, threads=BLOCK_SIZE) as (bx):
        tx = T.get_thread_binding()

        s_threshold_bin_id = T.alloc_shared([1], T.int32)
        s_histogram = T.alloc_shared([RADIX + 1], T.int32)
        s_num_input = T.alloc_shared([2], T.int32)
        s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)

        l_threshold_bin_id = T.alloc_var(T.int32)
        l_new_topk = T.alloc_var(T.int32)
        l_num_input = T.alloc_var(T.int32)
        l_bin_id32 = T.alloc_var(T.int32)
        l_val = T.alloc_var(T.int32)
        l_start_pos = T.alloc_var(T.int32)
        l_start_idx = T.alloc_var(T.int32)
        l_end_idx = T.alloc_var(T.int32)
        l_out_pos = T.alloc_var(T.int32)

        pos = T.alloc_var(T.int32)

        l_new_topk = topk
        l_start_idx = starts[bx]
        l_end_idx = ends[bx]

        # stage 1: use 8bit to do quick topk
        T.fill(s_histogram, 0)
        T.fill(s_num_input[0], 0)

        T.sync_threads()
        for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
            input_idx = s * BLOCK_SIZE + tx
            if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                inval_int16 = convert_to_uint16(input[bx, input_idx])
                T.atomic_add(s_histogram[inval_int16], 1)
        T.sync_threads()

        # cumsum
        if tx < RADIX:
            for i in T.serial(8):
                offset = 1 << i
                T.sync_threads(3, RADIX)
                if tx < RADIX - offset:
                    l_val = s_histogram[tx] + s_histogram[tx + offset]
                T.sync_threads(3, RADIX)
                if tx < RADIX - offset:
                    s_histogram[tx] = l_val

            # find threshold bin id
            T.sync_threads(3, RADIX)
            if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                s_threshold_bin_id[0] = tx
        T.sync_threads()
        l_threshold_bin_id = s_threshold_bin_id[0]
        l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
        T.sync_threads()

        # collect all elements with exponent ≥ threshold
        for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
            T.sync_threads()
            input_idx = s * BLOCK_SIZE + tx
            if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                bin_id = convert_to_uint16(input[bx, input_idx])
                l_bin_id32 = T.cast(bin_id, T.int32)
                if l_bin_id32 > l_threshold_bin_id:
                    # need a pos = T.atomic_add(s_histogram[bin_id32+1], 1)
                    pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                    index[bx, pos] = input_idx

                elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                    pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                    if pos < SMEM_INPUT_SIZE:
                        s_input_idx[0, pos] = input_idx
                    else:
                        T.atomic_add(overflow[bx], 1)

        # stage 2: tail pass
        for round in T.serial(4):
            if l_new_topk <= 0:
                break

            r_idx = round % 2
            l_start_pos = topk - l_new_topk

            T.sync_threads()
            T.fill(s_histogram, 0)
            if tx == 0:
                s_num_input[r_idx ^ 1] = 0
            T.sync_threads()

            l_num_input = T.min(s_num_input[r_idx], SMEM_INPUT_SIZE)
            for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                if s * BLOCK_SIZE + tx < l_num_input:
                    l_bin_id32 = T.cast(
                        ((convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >> (24 - round * 8)) & 0xFF), T.int32
                    )
                    T.atomic_add(s_histogram[l_bin_id32], 1)
            T.sync_threads()
            # cumsum
            if tx < RADIX:
                for i in T.serial(8):
                    offset = 1 << i
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        l_val = s_histogram[tx] + s_histogram[tx + offset]
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        s_histogram[tx] = l_val

                # find threshold bin id
                T.sync_threads(3, RADIX)
                if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                    s_threshold_bin_id[0] = tx
            T.sync_threads()

            l_threshold_bin_id = s_threshold_bin_id[0]
            l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
            T.sync_threads()

            for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                T.sync_threads()
                if s * BLOCK_SIZE + tx < l_num_input:
                    l_bin_id32 = T.cast(
                        ((convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >> (24 - round * 8)) & 0xFF), T.int32
                    )
                    if l_bin_id32 > l_threshold_bin_id:
                        pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                        index[bx, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                    elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                        if round == 3:
                            l_out_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                            if l_out_pos < topk:
                                index[bx, l_out_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                        else:
                            pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                            if pos < SMEM_INPUT_SIZE:
                                s_input_idx[r_idx ^ 1, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                            else:
                                T.atomic_add(overflow[bx], 1)


@tilelang.jit(pass_configs=pass_configs)
def tl_topk_from_hist_impl(input, hist, index, starts, ends, overflow, in_dtype=T.float32, out_dtype=T.int32):
    # Variant of tl_topk_impl for producers that already histogrammed the
    # logits while they still owned them (e.g. the fused prefill kernel's
    # selection-warp epilogue): stage 1's histogram build -- a full read of
    # `input` -- is replaced by loading the precomputed 256-bin histogram.
    # The collect pass and the stage-2 tail refinement are verbatim copies of
    # tl_topk_impl (a tilelang.jit function is a single kernel, so the shared
    # tail cannot be factored out); keep the two in sync.
    topk = T.const("topk")
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    RADIX = 1 << 8
    BLOCK_SIZE = 1024
    SMEM_INPUT_SIZE = 8192

    input: T.Tensor[(batch, seq_len), in_dtype]
    hist: T.Tensor[(batch, RADIX), out_dtype]
    index: T.Tensor[(batch, topk), out_dtype]
    starts: T.Tensor[(batch), out_dtype]
    ends: T.Tensor[(batch), out_dtype]
    overflow: T.Tensor[(batch), out_dtype]

    with T.Kernel(batch, threads=BLOCK_SIZE) as (bx):
        tx = T.get_thread_binding()

        s_threshold_bin_id = T.alloc_shared([1], T.int32)
        s_histogram = T.alloc_shared([RADIX + 1], T.int32)
        s_num_input = T.alloc_shared([2], T.int32)
        s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)

        l_threshold_bin_id = T.alloc_var(T.int32)
        l_new_topk = T.alloc_var(T.int32)
        l_num_input = T.alloc_var(T.int32)
        l_bin_id32 = T.alloc_var(T.int32)
        l_val = T.alloc_var(T.int32)
        l_start_pos = T.alloc_var(T.int32)
        l_start_idx = T.alloc_var(T.int32)
        l_end_idx = T.alloc_var(T.int32)
        l_out_pos = T.alloc_var(T.int32)

        pos = T.alloc_var(T.int32)

        l_new_topk = topk
        l_start_idx = starts[bx]
        l_end_idx = ends[bx]

        # stage 1: histogram comes from the producer; no full input read
        T.fill(s_histogram, 0)
        T.fill(s_num_input[0], 0)
        T.sync_threads()
        if tx < RADIX:
            s_histogram[tx] = hist[bx, tx]
        T.sync_threads()

        # cumsum
        if tx < RADIX:
            for i in T.serial(8):
                offset = 1 << i
                T.sync_threads(3, RADIX)
                if tx < RADIX - offset:
                    l_val = s_histogram[tx] + s_histogram[tx + offset]
                T.sync_threads(3, RADIX)
                if tx < RADIX - offset:
                    s_histogram[tx] = l_val

            # find threshold bin id
            T.sync_threads(3, RADIX)
            if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                s_threshold_bin_id[0] = tx
        T.sync_threads()
        l_threshold_bin_id = s_threshold_bin_id[0]
        l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
        T.sync_threads()

        # collect all elements with exponent ≥ threshold
        for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
            T.sync_threads()
            input_idx = s * BLOCK_SIZE + tx
            if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                bin_id = convert_to_uint16(input[bx, input_idx])
                l_bin_id32 = T.cast(bin_id, T.int32)
                if l_bin_id32 > l_threshold_bin_id:
                    pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                    index[bx, pos] = input_idx

                elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                    pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                    if pos < SMEM_INPUT_SIZE:
                        s_input_idx[0, pos] = input_idx
                    else:
                        T.atomic_add(overflow[bx], 1)

        # stage 2: tail pass
        for round in T.serial(4):
            if l_new_topk <= 0:
                break

            r_idx = round % 2
            l_start_pos = topk - l_new_topk

            T.sync_threads()
            T.fill(s_histogram, 0)
            if tx == 0:
                s_num_input[r_idx ^ 1] = 0
            T.sync_threads()

            l_num_input = T.min(s_num_input[r_idx], SMEM_INPUT_SIZE)
            for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                if s * BLOCK_SIZE + tx < l_num_input:
                    l_bin_id32 = T.cast(
                        ((convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >> (24 - round * 8)) & 0xFF), T.int32
                    )
                    T.atomic_add(s_histogram[l_bin_id32], 1)
            T.sync_threads()
            # cumsum
            if tx < RADIX:
                for i in T.serial(8):
                    offset = 1 << i
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        l_val = s_histogram[tx] + s_histogram[tx + offset]
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        s_histogram[tx] = l_val

                # find threshold bin id
                T.sync_threads(3, RADIX)
                if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                    s_threshold_bin_id[0] = tx
            T.sync_threads()

            l_threshold_bin_id = s_threshold_bin_id[0]
            l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
            T.sync_threads()

            for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                T.sync_threads()
                if s * BLOCK_SIZE + tx < l_num_input:
                    l_bin_id32 = T.cast(
                        ((convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >> (24 - round * 8)) & 0xFF), T.int32
                    )
                    if l_bin_id32 > l_threshold_bin_id:
                        pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                        index[bx, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                    elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                        if round == 3:
                            l_out_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                            if l_out_pos < topk:
                                index[bx, l_out_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                        else:
                            pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                            if pos < SMEM_INPUT_SIZE:
                                s_input_idx[r_idx ^ 1, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                            else:
                                T.atomic_add(overflow[bx], 1)


def tl_topk(input, starts, ends, topk, strict=True, hist=None):
    """Radix top-k over `input[b, starts[b]:ends[b]]`.

    Precondition: every row's window must hold MORE than `topk` candidates
    (ends[b] - starts[b] > topk). With exactly topk or fewer, the
    threshold-bin search never fires and the selector reads an
    uninitialized bin id (garbage indices / illegal access).

    If `hist` is given -- a (batch, 256) int32 histogram of
    `convert_to_uint16(input)` over exactly the [starts, ends) window,
    typically built by the producing kernel while the logits were still
    in registers -- stage 1 skips its full read of `input`.
    """
    batch, seq_len = input.shape
    indexes = torch.zeros(batch, topk, dtype=torch.int32, device=input.device)
    overflow = torch.zeros(batch, dtype=torch.int32, device=input.device)
    if hist is None:
        tl_topk_impl(input, indexes, starts, ends, overflow)
    else:
        assert hist.shape == (batch, 256) and hist.dtype == torch.int32, (
            f"hist must be (batch, 256) int32, got {tuple(hist.shape)} {hist.dtype}"
        )
        tl_topk_from_hist_impl(input, hist, indexes, starts, ends, overflow)
    if strict:
        dropped = int(overflow.sum())
        if dropped:
            raise RuntimeError(
                f"tl_topk threshold-bucket overflow: {dropped} candidate(s) exceeded "
                f"the 8192-slot buffer; the returned top-k would be approximate. "
                f"Pass strict=False to accept approximate selection."
            )
    return indexes


def test_topk_selector(batch=64, seq_len=32 * 1024, topk=2048):
    torch.manual_seed(1)
    input = torch.randn(batch, seq_len, dtype=torch.float32).cuda()
    starts = torch.zeros(batch, dtype=torch.int32).cuda()
    ends = torch.ones(batch, dtype=torch.int32).cuda() * seq_len

    indexes = tl_topk(input, starts, ends, topk)
    print(indexes)

    indexes_ref = torch.topk(input, topk, dim=-1)[1]
    print(indexes_ref)

    # indexes_ref = fast_topk(input, topk)
    # print(indexes_ref)

    # Calculate intersection of out_ref and out_trt
    for i in range(batch):
        ref_np = indexes_ref[i].cpu().to(torch.int32).numpy()
        trt_np = indexes[i].cpu().to(torch.int32).numpy()

        set_ref = set(ref_np)
        set_trt = set(trt_np)
        intersection = set_ref & set_trt
        print("selected/all:", len(intersection), "/", len(set_ref), "=", len(intersection) / len(set_ref))

    # Performance test with CUDA events

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        _ = tl_topk(input, starts, ends, topk)
    torch.cuda.synchronize()

    n_iters = 20
    start_event.record()
    for _ in range(n_iters):
        _ = tl_topk(input, starts, ends, topk)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Average tl_topk time: {elapsed_time_ms / n_iters:.3f} ms")

    # Torch topk time
    start_event.record()
    for _ in range(n_iters):
        _ = torch.topk(input, topk, dim=-1)[1]
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Average torch.topk time: {elapsed_time_ms / n_iters:.3f} ms")


def run_regression_perf(batch=64, seq_len=32 * 1024, topk=2048):
    torch.manual_seed(1)
    input = torch.randn(batch, seq_len, dtype=torch.float32).cuda()
    starts = torch.zeros(batch, dtype=torch.int32).cuda()
    ends = torch.ones(batch, dtype=torch.int32).cuda() * seq_len

    from tilelang.profiler import do_bench

    def run_kernel_only():
        tl_topk(input, starts, ends, topk)

    return do_bench(run_kernel_only, backend="cupti")


if __name__ == "__main__":
    test_topk_selector()
