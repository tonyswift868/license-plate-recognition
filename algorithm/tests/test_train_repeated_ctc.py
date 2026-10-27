"""CPU-only algebra checks for compressed repeated-sample training batches."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _repeat_flat_targets(targets: torch.Tensor, lengths: torch.Tensor,
                         repeats: torch.Tensor) -> torch.Tensor:
    offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    pieces = [targets[offsets[index]:offsets[index + 1]] for index in repeats.tolist()]
    return torch.cat(pieces)


def test_counted_unreduced_ctc_matches_expanded_mean_and_gradients():
    torch.manual_seed(7)
    blank, time_steps, classes = 5, 7, 6
    targets = torch.tensor([0, 1, 2, 3, 1, 4, 0], dtype=torch.long)
    target_lengths = torch.tensor([2, 3, 2], dtype=torch.long)
    input_lengths = torch.tensor([time_steps, time_steps, time_steps], dtype=torch.long)
    counts = torch.tensor([3, 1, 2], dtype=torch.long)
    expanded_indices = torch.arange(len(counts)).repeat_interleave(counts)
    expanded_targets = _repeat_flat_targets(targets, target_lengths, expanded_indices)

    unique_logits = torch.randn(time_steps, len(counts), classes, requires_grad=True)
    expanded_logits = unique_logits.index_select(1, expanded_indices)
    expanded_loss = nn.CTCLoss(blank=blank, reduction="mean")(
        expanded_logits.log_softmax(2), expanded_targets,
        input_lengths.index_select(0, expanded_indices),
        target_lengths.index_select(0, expanded_indices))
    expanded_gradient, = torch.autograd.grad(expanded_loss, unique_logits)

    counted_logits = unique_logits.detach().clone().requires_grad_(True)
    per_unique_loss = nn.CTCLoss(blank=blank, reduction="none")(
        counted_logits.log_softmax(2), targets, input_lengths, target_lengths)
    counted_loss = ((per_unique_loss / target_lengths) * counts).sum() / counts.sum()
    counted_gradient, = torch.autograd.grad(counted_loss, counted_logits)

    torch.testing.assert_close(counted_loss, expanded_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(counted_gradient, expanded_gradient, rtol=1e-6, atol=1e-7)


def test_counted_geometry_mean_matches_expanded_mean_and_gradients():
    torch.manual_seed(11)
    counts = torch.tensor([3, 1, 2], dtype=torch.long)
    expanded_indices = torch.arange(len(counts)).repeat_interleave(counts)
    target = torch.randn(3, 4)

    prediction = torch.randn(3, 4, requires_grad=True)
    expanded_loss = F.smooth_l1_loss(
        prediction.index_select(0, expanded_indices), target.index_select(0, expanded_indices),
        reduction="mean")
    expanded_gradient, = torch.autograd.grad(expanded_loss, prediction)

    counted_prediction = prediction.detach().clone().requires_grad_(True)
    per_unique = F.smooth_l1_loss(counted_prediction, target, reduction="none").mean(dim=1)
    counted_loss = (per_unique * counts).sum() / counts.sum()
    counted_gradient, = torch.autograd.grad(counted_loss, counted_prediction)

    torch.testing.assert_close(counted_loss, expanded_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(counted_gradient, expanded_gradient, rtol=1e-6, atol=1e-7)
