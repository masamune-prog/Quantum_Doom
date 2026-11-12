"""WideCircuitPolicy must reproduce the PennyLane QuantumE1M1Policy at 12 qubits."""

import torch

from e1m1_quantum_model import QuantumE1M1Policy
from wide_circuit import WideCircuitPolicy


def check(entangle, n_layers=2):
    torch.manual_seed(0)
    reference = QuantumE1M1Policy(n_layers=n_layers, n_actions=7, entangle=entangle)
    with torch.no_grad():
        reference.theta.normal_(0, 1.0)
        reference.input_scale.uniform_(0.3, 1.2)
    wide = WideCircuitPolicy(n_qubits=12, n_layers=n_layers, entangle=entangle)
    wide.load_state_dict(reference.state_dict(), strict=False)
    x = (torch.rand(5, 12) * 2 - 1) * torch.pi
    ref_logits, ref_value = reference(x)
    logits, value = wide(x)
    assert torch.allclose(logits, ref_logits.float(), atol=1e-4), (logits, ref_logits)
    assert torch.allclose(value, ref_value.float(), atol=1e-4)
    # gradients agree too
    ref_logits.float().sum().backward()
    wide.train()
    logits, _ = wide(x)
    logits.sum().backward()
    assert torch.allclose(wide.theta.grad, reference.theta.grad.float(), atol=1e-3)
    assert torch.allclose(wide.input_scale.grad, reference.input_scale.grad.float(), atol=1e-3)


def test_matches_pennylane():
    check(entangle=True)
    check(entangle=False)
    check(entangle=True, n_layers=4)


def test_param_count():
    assert sum(p.numel() for p in WideCircuitPolicy(16, 23).parameters()) == 1106


if __name__ == "__main__":
    test_matches_pennylane()
    test_param_count()
    print("ok")
