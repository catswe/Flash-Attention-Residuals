import torch
import triton
from torch.library import triton_op, wrap_triton

from ..kernels import phase_1
from ..kernels.reduce import reduce_grad_queries_kernel

EPS = torch.finfo(torch.float32).eps


@triton_op(
    "flash_attn_res::_phase_1_batched_attention_forward_with_aux",
    mutates_args={},
)
def _phase_1_batched_attention_forward_with_aux_triton_op(
    block_representations: torch.Tensor,
    pseudo_queries: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_source_blocks = block_representations.shape[0]
    num_queries = pseudo_queries.shape[0]
    B, T, D = block_representations.shape[1:]
    BT = B * T

    softmax_outputs = torch.empty(
        (num_queries, B, T, D),
        device=block_representations.device,
        dtype=torch.bfloat16,
    )

    lses = torch.empty(
        (num_queries, B, T),
        device=block_representations.device,
        dtype=torch.float32,
    )

    inverse_rms_norms = torch.empty(
        (B, T, num_source_blocks),
        device=block_representations.device,
        dtype=torch.float32,
    )

    attention_logits = torch.empty(
        (num_queries, B, T, num_source_blocks),
        device=block_representations.device,
        dtype=torch.float32,
    )

    wrap_triton(phase_1.phase_1_batched_attention_forward_kernel)[(BT,)](
        block_representations,
        pseudo_queries,
        softmax_outputs,
        lses,
        inverse_rms_norms,
        attention_logits,
        eps,
        num_source_blocks,
        BT,
        D,
        num_queries,
        triton.next_power_of_2(num_source_blocks),
    )

    return softmax_outputs, lses, inverse_rms_norms, attention_logits


def phase_1_batched_attention_triton_op(
    block_representations: torch.Tensor,
    pseudo_queries: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    softmax_outputs, lses, _inverse_rms_norms, _attention_logits = (
        _phase_1_batched_attention_forward_with_aux_triton_op(
            block_representations,
            pseudo_queries,
            eps,
        )
    )
    return softmax_outputs, lses


@triton_op(
    "flash_attn_res::_phase_1_batched_attention_backward",
    mutates_args={},
)
def _batched_attention_backward_triton_op(
    block_representations: torch.Tensor,
    pseudo_queries: torch.Tensor,
    lses: torch.Tensor,
    inverse_rms_norms: torch.Tensor,
    attention_logits: torch.Tensor,
    grad_softmax_outputs: torch.Tensor,
    grad_lses: torch.Tensor,
    has_grad_lses: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_source_blocks = block_representations.shape[0]
    num_queries = pseudo_queries.shape[0]
    B, T, D = block_representations.shape[1:]

    grad_block_representations = torch.empty(
        (num_source_blocks, B, T, D),
        device=block_representations.device,
        dtype=torch.float32,
    )

    grad_pseudo_queries = torch.zeros(
        (num_queries, D),
        device=pseudo_queries.device,
        dtype=torch.float32,
    )

    grad_pseudo_queries_partial = torch.empty(
        (num_queries, B, T, D),
        device=pseudo_queries.device,
        dtype=torch.float32,
    )

    _batched_attention_backward_accumulate(
        block_representations,
        pseudo_queries,
        lses,
        grad_softmax_outputs,
        grad_lses if has_grad_lses else None,
        grad_block_representations,
        grad_pseudo_queries,
        grad_pseudo_queries_partial,
        eps,
        False,
        inverse_rms_norms,
        attention_logits,
    )

    return grad_block_representations, grad_pseudo_queries


def _batched_attention_backward_accumulate(
    block_representations,
    pseudo_queries,
    lses,
    grad_softmax_outputs,
    grad_lses,
    grad_block_representations,
    grad_pseudo_queries,
    grad_pseudo_queries_partial,
    eps,
    accumulate_grad_blocks,
    inverse_rms_norms,
    attention_logits,
) -> None:
    num_source_blocks = block_representations.shape[0]
    num_queries = pseudo_queries.shape[0]
    B, T, D = block_representations.shape[1:]
    BT = B * T

    has_grad_lses = grad_lses is not None

    if grad_lses is None:
        grad_lses = lses

    wrap_triton(phase_1.phase_1_batched_attention_backward_kernel)[(BT,)](
        block_representations,
        pseudo_queries,
        lses,
        inverse_rms_norms,
        attention_logits,
        grad_softmax_outputs,
        grad_lses,
        grad_block_representations,
        grad_pseudo_queries_partial,
        eps,
        num_source_blocks,
        BT,
        D,
        num_queries,
        triton.next_power_of_2(num_source_blocks),
        has_grad_lses,
        accumulate_grad_blocks,
    )

    wrap_triton(reduce_grad_queries_kernel)[
        lambda META: (
            triton.cdiv(BT, META["BLOCK_BATCH_SEQ"]),
            num_queries,
            triton.cdiv(D, META["BLOCK_HIDDEN"]),
        )
    ](
        grad_pseudo_queries_partial,
        grad_pseudo_queries,
        BT,
        D,
    )


def setup_context(ctx, inputs, output):
    block_representations, pseudo_queries, eps = inputs
    _softmax_outputs, lses, inverse_rms_norms, attention_logits = output

    ctx.save_for_backward(
        block_representations,
        pseudo_queries,
        lses,
        inverse_rms_norms,
        attention_logits,
    )
    ctx.eps = eps


def backward(
    ctx,
    grad_softmax_outputs,
    grad_lses,
    _grad_inverse_rms_norms,
    _grad_attention_logits,
):
    (
        block_representations,
        pseudo_queries,
        lses,
        inverse_rms_norms,
        attention_logits,
    ) = ctx.saved_tensors

    num_queries = pseudo_queries.shape[0]
    B, T, D = block_representations.shape[1:]

    if grad_softmax_outputs is None:
        grad_softmax_outputs = torch.zeros(
            (num_queries, B, T, D),
            device=block_representations.device,
            dtype=torch.float32,
        )
    else:
        grad_softmax_outputs = grad_softmax_outputs.contiguous()

    has_grad_lses = grad_lses is not None

    if grad_lses is None:
        grad_lses = lses
    else:
        grad_lses = grad_lses.contiguous()

    grad_block_representations, grad_pseudo_queries = (
        _batched_attention_backward_triton_op(
            block_representations,
            pseudo_queries,
            lses,
            inverse_rms_norms,
            attention_logits,
            grad_softmax_outputs,
            grad_lses,
            has_grad_lses,
            ctx.eps,
        )
    )

    return (
        (
            grad_block_representations.to(block_representations.dtype)
            if ctx.needs_input_grad[0]
            else None
        ),
        (
            grad_pseudo_queries.to(pseudo_queries.dtype)
            if ctx.needs_input_grad[1]
            else None
        ),
        None,
    )


_phase_1_batched_attention_forward_with_aux_triton_op.register_autograd(
    backward,
    setup_context=setup_context,
)
