"""Wide/deep version of the E1M1 circuit head, simulated directly in torch.

Same gate set as QuantumE1M1Policy: per layer, every wire gets RY(x * s), RZ(theta0),
RX(theta1), then a CNOT ring. Wire w encodes input x[w mod n_inputs], so extra qubits
re-upload the 12 frozen angles with their own learned scales. <Z> on wires 0..A-1 are the
action logits and <Z> on wire A is the value, each times a learned gain.

Parameters: 3 * n_qubits * n_layers + 2. PennyLane default.qubit is far too slow at
16-20 qubits and ~20 layers, so this is a batched statevector simulator with per-layer
gradient checkpointing. test_wide_circuit.py checks it against PennyLane at 12 qubits.
"""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.utils.checkpoint import checkpoint


def _cnot_ring_permutation(n_qubits):
    """Gather index so that state[..., perm] applies CNOT(0,1)...CNOT(n-2,n-1), CNOT(n-1,0).

    Wire 0 is the most significant bit (PennyLane convention).
    """
    index = np.arange(2 ** n_qubits)
    bit = lambda w: 1 << (n_qubits - 1 - w)
    perm = index.copy()
    for control in range(n_qubits):
        target = (control + 1) % n_qubits
        flip = np.where(index & bit(control), index ^ bit(target), index)
        perm = perm[flip]
    return torch.as_tensor(perm)


class WideCircuitPolicy(nn.Module):
    GROUP = 4
    def __init__(self, n_qubits=16, n_layers=23, n_inputs=12, n_actions=7,
                 input_scale_init=0.5, entangle=True, use_checkpoint=True):
        super().__init__()
        if n_actions + 1 > n_qubits:
            raise ValueError("Need one measured qubit per action plus one for value")
        self.n_qubits, self.n_layers = n_qubits, n_layers
        self.n_inputs, self.n_actions = n_inputs, n_actions
        self.entangle, self.use_checkpoint = entangle, use_checkpoint
        self.theta = nn.Parameter(0.05 * torch.randn(n_layers, n_qubits, 2))
        self.input_scale = nn.Parameter(torch.full((n_layers, n_qubits), float(input_scale_init)))
        self.logit_gain = nn.Parameter(torch.tensor(2.0))
        self.value_gain = nn.Parameter(torch.tensor(5.0))
        self.register_buffer("wire_input", torch.arange(n_qubits) % n_inputs, persistent=False)
        self.register_buffer("ring", _cnot_ring_permutation(n_qubits), persistent=False)
        signs = 1 - 2 * ((torch.arange(2 ** n_qubits)[:, None]
                          >> (n_qubits - 1 - torch.arange(n_actions + 1))) & 1)
        self.register_buffer("z_signs", signs.float(), persistent=False)

    def _layer(self, state, angles, theta):
        """angles: (B, n) RY angles; theta: (n, 2) RZ/RX angles. state: (B, 2**n) complex."""
        n, batch = self.n_qubits, state.shape[0]
        hy = angles / 2
        cy, sy = torch.cos(hy), torch.sin(hy)
        ez = torch.polar(torch.ones_like(theta[:, 0]), -theta[:, 0] / 2)  # e^{-i b/2}
        cx, sx = torch.cos(theta[:, 1] / 2), torch.sin(theta[:, 1] / 2)
        # U = RX(g) RZ(b) RY(a); RZ RY = [[ez c, -ez s], [conj(ez) s, conj(ez) c]]
        a00, a01 = ez * cy, -ez * sy
        a10, a11 = ez.conj() * sy, ez.conj() * cy
        isx = 1j * sx
        u = torch.stack((torch.stack((cx * a00 - isx * a10, cx * a01 - isx * a11), -1),
                         torch.stack((-isx * a00 + cx * a10, -isx * a01 + cx * a11), -1)), -2)
        # u: (B, n, 2, 2). Fuse GROUP adjacent wires into one 2^k x 2^k Kronecker block and
        # apply it to the leading wires with bmm, then rotate those wires to the back. After
        # all groups the wire order is restored. ~8x faster than one einsum per wire.
        start = 0
        while start < n:
            k = min(self.GROUP, n - start)
            block = u[:, start]
            for wire in range(start + 1, start + k):
                size = block.shape[-1] * 2
                block = torch.einsum("bij,bkl->bikjl", block, u[:, wire]).reshape(batch, size, size)
            state = torch.bmm(block, state.reshape(batch, 2 ** k, -1))
            state = state.transpose(1, 2).reshape(batch, 2 ** n)
            start += k
        if self.entangle:
            state = state.index_select(1, self.ring)
        return state

    def forward(self, obs):
        obs = obs.float()
        batch = obs.shape[0]
        x = obs[:, self.wire_input]  # (B, n)
        state = torch.zeros(batch, 2 ** self.n_qubits, dtype=torch.complex64, device=obs.device)
        state[:, 0] = 1
        for layer in range(self.n_layers):
            angles = x * self.input_scale[layer]
            if self.use_checkpoint and torch.is_grad_enabled():
                state = checkpoint(self._layer, state, angles, self.theta[layer],
                                   use_reentrant=False)
            else:
                state = self._layer(state, angles, self.theta[layer])
        probs = state.real ** 2 + state.imag ** 2
        measured = probs @ self.z_signs  # (B, A+1)
        return (self.logit_gain * measured[:, :self.n_actions],
                self.value_gain * measured[:, self.n_actions])

    def act(self, obs, action=None):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
