import triton
import triton.language as tl

reduce_configs = [
    triton.Config(
        {
            "BLOCK_BATCH_SEQ": block_batch_seq,
            "BLOCK_HIDDEN": block_hidden,
        },
        num_warps=num_warps,
        num_stages=1,
    )
    for block_batch_seq in [128, 256]
    for block_hidden in [32, 64]
    for num_warps in [4, 8]
]


@triton.autotune(
    configs=reduce_configs,
    key=["NUM_BATCH_SEQ", "HIDDEN_DIM"],
    restore_value=[
        "grad_pseudo_query_accumulator_ptr",
    ],
)
def reduce_grad_queries_kernel(
    grad_partial_ptr,
    grad_accumulator_ptr,
    NUM_BATCH_SEQ: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    NUM_QUERIES: tl.constexpr,
    BLOCK_BATCH_SEQ: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    batch_seq_block_idx = tl.program_id(0)
    query_idx = tl.program_id(1)
    hidden_block_idx = tl.program_id(2)

    batch_seq_offsets = batch_seq_block_idx * BLOCK_BATCH_SEQ + tl.arange(
        0, BLOCK_BATCH_SEQ
    )
    hidden_offsets = hidden_block_idx * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)

    grad_tile = tl.load(
        grad_partial_ptr
        + query_idx * NUM_BATCH_SEQ * HIDDEN_DIM
        + batch_seq_offsets[:, None] * HIDDEN_DIM
        + hidden_offsets[None, :],
        mask=(
            (batch_seq_offsets[:, None] < NUM_BATCH_SEQ)
            & (hidden_offsets[None, :] < HIDDEN_DIM)
        ),
        other=0.0,
    ).to(tl.float32)

    grad_reduced = tl.sum(grad_tile, axis=0)

    tl.atomic_add(
        grad_accumulator_ptr + query_idx * HIDDEN_DIM + hidden_offsets,
        grad_reduced,
        mask=hidden_offsets < HIDDEN_DIM,
        sem="relaxed",
    )
