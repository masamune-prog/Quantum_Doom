"""Draw the 12-qubit, 2-layer E1M1 circuit from the real model code (PennyLane qml.draw / draw_mpl).

Gate parameters are hidden (decimals=None) so the diagram shows the structure: per layer,
RY(s*z_i) -> RZ -> RX on every qubit, then the CNOT ring 0->1->...->11->0. The Pauli-Z
expectations of qubits 0-7 are the 7 action logits and the value (qubits 8-11 are unmeasured).
"""

import sys

import matplotlib

matplotlib.use("Agg")
import pennylane as qml
import torch

from e1m1_quantum_model import QuantumE1M1Policy

out = sys.argv[1] if len(sys.argv) > 1 else "circuit_diagram"
model = QuantumE1M1Policy(n_qubits=12, n_layers=2, n_actions=7)
x = torch.linspace(-3.0, 3.0, 12)  # example bottleneck values; they do not change the structure
args = (x, model.theta.detach(), model.input_scale.detach())

text = qml.draw(model.circuit, decimals=None, show_all_wires=True)(*args)
print(text)
with open(f"{out}/circuit_e1m1.txt", "w") as handle:
    handle.write(text + "\n")

for style, name in (("black_white", "circuit_e1m1"), ("pennylane", "circuit_e1m1_color")):
    fig, ax = qml.draw_mpl(model.circuit, decimals=None, show_all_wires=True, style=style,
                           fontsize=11)(*args)
    fig.savefig(f"{out}/{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{out}/{name}.pdf", bbox_inches="tight")
    print("saved", name, fig.get_size_inches())
