"""Circuit policy and circuit value estimate for E1M1.

The seven action logits and the value estimate are direct Pauli-Z expectation
values. There is no learned classical feature extractor, MLP, or linear head.
The two scalar gains only set the output units.
"""

import pennylane as qml
import torch
from torch import nn
from torch.distributions import Categorical


class QuantumE1M1Policy(nn.Module):
    def __init__(self, n_qubits=12, n_layers=2, n_actions=7, memory_bridges=False):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.memory_bridges = memory_bridges
        if n_actions + 1 > n_qubits:
            raise ValueError("Need one measured qubit per action plus one for value")
        if memory_bridges and n_qubits != 12:
            raise ValueError("Memory bridges require the 12-feature observation layout")
        self.theta = nn.Parameter(0.05 * torch.randn(n_layers, n_qubits, 2))
        self.input_scale = nn.Parameter(torch.ones(n_layers, n_qubits))
        self.logit_gain = nn.Parameter(torch.tensor(2.0))
        self.value_gain = nn.Parameter(torch.tensor(5.0))
        self.bridge_theta = (nn.Parameter(0.35 + 0.05 * torch.randn(n_layers, 4, n_actions + 1))
                             if memory_bridges else None)
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x, theta, input_scale, bridge_theta=None):
            for layer in range(n_layers):
                for wire in range(n_qubits):
                    qml.RY(x[..., wire] * input_scale[layer, wire], wires=wire)
                    qml.RZ(theta[layer, wire, 0], wires=wire)
                    qml.RX(theta[layer, wire, 1], wires=wire)
                for wire in range(n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
                if memory_bridges:
                    for memory_wire in range(8, 12):
                        for output_wire in range(n_actions + 1):
                            qml.CRY(bridge_theta[layer, memory_wire - 8, output_wire],
                                    wires=[memory_wire, output_wire])
            return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(n_actions + 1))

        self.circuit = circuit

    def forward(self, obs):
        if self.memory_bridges:
            measured = self.circuit(obs, self.theta, self.input_scale, self.bridge_theta)
        else:
            measured = self.circuit(obs, self.theta, self.input_scale)
        measured = torch.stack(measured, dim=-1).float()
        return (self.logit_gain * measured[..., :self.n_actions],
                self.value_gain * measured[..., self.n_actions])

    def act(self, obs, action=None):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
