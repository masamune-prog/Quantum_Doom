import pennylane as qml
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class DataReuploadingPQC(nn.Module):
    def __init__(self, n_qubits=8, n_layers=4, n_actions=3, entangle=True, noise_p=0.0):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_actions = n_actions
        self.entangle = entangle
        self.noise_p = noise_p

        dev_name = "default.mixed" if noise_p > 0 else "default.qubit"
        self.dev = qml.device(dev_name, wires=n_qubits)

        # Variational gate angles (L, N, 2) and trainable input frequency scales (L, N)
        self.theta = nn.Parameter(torch.empty(n_layers, n_qubits, 2).uniform_(-0.1, 0.1))
        self.lmbda = nn.Parameter(torch.ones(n_layers, n_qubits))
        # Trainable linear readout mapping all N qubit expectation values to action logits
        self.head = nn.Linear(n_qubits, n_actions)
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.constant_(self.head.bias, 0.0)

        diff_method = "backprop"
        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def _circuit(inputs, theta, lmbda):
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RX(inputs[..., i] * lmbda[l, i], wires=i)
                    qml.RY(theta[l, i, 0], wires=i)
                    qml.RZ(theta[l, i, 1], wires=i)
                if self.entangle:
                    for i in range(self.n_qubits):
                        qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                        if self.noise_p > 0:
                            qml.DepolarizingChannel(self.noise_p, wires=(i + 1) % self.n_qubits)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.qnode = _circuit

    def get_all_expvals(self, x: torch.Tensor) -> torch.Tensor:
        expvals = self.qnode(x, self.theta, self.lmbda)
        return torch.stack(expvals, dim=-1).to(dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expvals = self.get_all_expvals(x)
        return self.head(expvals)


class DoomActorCritic(nn.Module):
    def __init__(self, n_qubits=8, n_layers=4, n_actions=3, mode="quantum_entangled", noise_p=0.0):
        super().__init__()
        self.mode = mode
        if mode == "quantum_entangled":
            self.actor = DataReuploadingPQC(n_qubits, n_layers, n_actions, entangle=True, noise_p=noise_p)
        elif mode == "quantum_product":
            self.actor = DataReuploadingPQC(n_qubits, n_layers, n_actions, entangle=False, noise_p=noise_p)
        elif mode == "classical_mlp":
            # Parameter-matched classical baseline (8*14 + 14 + 14*3 + 3 = 171 params vs 195 quantum params)
            self.actor = nn.Sequential(
                nn.Linear(n_qubits, 14),
                nn.Tanh(),
                nn.Linear(14, n_actions)
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.critic = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
