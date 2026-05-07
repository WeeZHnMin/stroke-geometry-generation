from __future__ import annotations

import torch
def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(pred.dtype)
    sq = (pred - target) ** 2 * mask_f
    denom = (mask_f.sum() * pred.size(-1)).clamp_min(1.0)
    return sq.sum() / denom


def compute_diffusion_reconstruction_loss(
    *,
    pred_steps: torch.Tensor,
    target_steps: torch.Tensor,
    pen_targets: torch.Tensor,
    seq_mask: torch.Tensor,
    lambda_pen: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_xy = pred_steps[..., :2]
    target_xy = target_steps[..., :2]

    loss_xy = masked_mse(pred_xy, target_xy, seq_mask)

    pred_pen = pred_steps[..., 2]
    target_pen = pen_targets.to(pred_pen.dtype)
    loss_pen = masked_mse(pred_pen.unsqueeze(-1), target_pen.unsqueeze(-1), seq_mask)

    loss = loss_xy + lambda_pen * loss_pen

    with torch.no_grad():
        valid = seq_mask.unsqueeze(-1).expand_as(target_xy)
        xy_mae = (pred_xy - target_xy).abs()[valid].mean()
        x_mae = (pred_xy[..., 0] - target_xy[..., 0]).abs()[seq_mask].mean()
        y_mae = (pred_xy[..., 1] - target_xy[..., 1]).abs()[seq_mask].mean()
        pred_pen_ids = pred_pen.round().clamp(min=0, max=2).long()
        pen_acc = ((pred_pen_ids == pen_targets) & seq_mask).float().sum() / seq_mask.float().sum().clamp_min(1.0)
        pen_mae = (pred_pen - target_pen).abs()[seq_mask].mean()

    metrics = {
        "loss": float(loss.item()),
        "loss_xy": float(loss_xy.item()),
        "loss_pen": float(loss_pen.item()),
        "xy_mae": float(xy_mae.item()),
        "x_mae": float(x_mae.item()),
        "y_mae": float(y_mae.item()),
        "pen_acc": float(pen_acc.item()),
        "pen_mae": float(pen_mae.item()),
    }
    return loss, metrics
