from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(pred.dtype)
    sq = (pred - target) ** 2 * mask_f
    denom = (mask_f.sum() * pred.size(-1)).clamp_min(1.0)
    return sq.sum() / denom


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    ignore_targets = flat_targets.masked_fill(~flat_mask, -100)
    return F.cross_entropy(flat_logits, ignore_targets, ignore_index=-100)


def onehot_pen_to_ids(steps: torch.Tensor) -> torch.Tensor:
    return steps[..., 2:].argmax(dim=-1)


def first_end_index(pen_ids: torch.Tensor, seq_mask: torch.Tensor, end_id: int = 2) -> torch.Tensor:
    """Return first end_all index for each sample, or last valid index if missing."""
    batch, seq_len = pen_ids.shape
    valid_positions = torch.arange(seq_len, device=pen_ids.device).unsqueeze(0).expand(batch, -1)
    end_mask = (pen_ids == end_id) & seq_mask
    fallback = seq_mask.long().sum(dim=1).clamp_min(1) - 1
    first_idx = torch.where(
        end_mask.any(dim=1),
        end_mask.float().argmax(dim=1),
        fallback,
    )
    return first_idx


def compute_absolute_positions(pred_deltas: torch.Tensor, start_pos: torch.Tensor) -> torch.Tensor:
    return start_pos + torch.cumsum(pred_deltas, dim=1)


def compute_diffusion_reconstruction_loss(
    *,
    pred_steps: torch.Tensor,
    target_steps: torch.Tensor,
    target_abs: torch.Tensor,
    start_pos: torch.Tensor,
    seq_mask: torch.Tensor,
    lambda_abs: float = 1.0,
    lambda_pen: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_deltas = pred_steps[..., :2]
    target_deltas = target_steps[..., :2]

    loss_rel = masked_mse(pred_deltas, target_deltas, seq_mask)
    pred_abs = compute_absolute_positions(pred_deltas, start_pos)
    loss_abs = masked_mse(pred_abs, target_abs, seq_mask)

    pen_logits = pred_steps[..., 2:]
    pen_targets = onehot_pen_to_ids(target_steps)
    loss_pen = masked_cross_entropy(pen_logits, pen_targets, seq_mask)

    loss = loss_rel + lambda_abs * loss_abs + lambda_pen * loss_pen

    with torch.no_grad():
        valid = seq_mask.unsqueeze(-1).expand_as(target_deltas)
        delta_mae = (pred_deltas - target_deltas).abs()[valid].mean()
        abs_mae = (pred_abs - target_abs).abs()[valid].mean()
        pred_pen_ids = pen_logits.argmax(dim=-1)
        pen_acc = ((pred_pen_ids == pen_targets) & seq_mask).float().sum() / seq_mask.float().sum().clamp_min(1.0)
        pred_end_idx = first_end_index(pred_pen_ids, seq_mask)
        true_end_idx = first_end_index(pen_targets, seq_mask)
        end_pos_error = (pred_end_idx - true_end_idx).abs().float().mean()
        pred_length = pred_end_idx.float() + 1.0
        true_length = true_end_idx.float() + 1.0
        length_error = (pred_length - true_length).abs().mean()

    metrics = {
        "loss": float(loss.item()),
        "loss_rel": float(loss_rel.item()),
        "loss_abs": float(loss_abs.item()),
        "loss_pen": float(loss_pen.item()),
        "delta_mae": float(delta_mae.item()),
        "abs_mae": float(abs_mae.item()),
        "pen_acc": float(pen_acc.item()),
        "end_pos_error": float(end_pos_error.item()),
        "length_error": float(length_error.item()),
    }
    return loss, metrics
