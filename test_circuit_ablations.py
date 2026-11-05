"""Check the two circuit ablations behave as intended (run on CPU via Slurm)."""

import torch

from pixel_nav_models import PixelNavPolicy

torch.manual_seed(0)
z = torch.rand(4, 12) * 2 - 1
for head in ("quantum", "quantum_noent", "quantum_frozen"):
    model = PixelNavPolicy(head=head, n_actions=7, aux_size=20, aux_route=3)
    trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.head.parameters())
    logits, value = model.head(z)
    z2 = z.clone()
    z2[:, 5] += 0.7  # change only bottleneck value 5
    logits2, value2 = model.head(z2)
    moved = ((logits2 - logits).abs().max(0).values > 1e-6).tolist()
    print(f"{head:15s} head params {total}, trainable {trainable}; logits that change when only "
          f"z[5] changes: {[k for k, m in enumerate(moved) if m]}")
