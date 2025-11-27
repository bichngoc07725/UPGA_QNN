# Q_PGA_models.py
import torch
import torch.nn as nn
import pennylane as qml
from utility import *

# ============================== QUANTUM DEVICE ==============================
n_qubits = 8
dev = qml.device("default.qubit.torch", wires=n_qubits, shots=None)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    # Angle embedding
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
        qml.RZ(inputs[i + n_qubits], wires=i)

    # 3-layer hardware-efficient ansatz
    for layer in range(3):
        for i in range(n_qubits):
            qml.Rot(*weights[layer, i], wires=i)
        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])
        qml.CNOT(wires=[n_qubits-1, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumBlock(nn.Module):
    def __init__(self, in_features=64, out_features=128):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(in_features, 32), nn.Tanh(),
            nn.Linear(32, n_qubits*2)
        )
        self.qnode = qml.qnn.TorchLayer(quantum_circuit, {"weights": (3, n_qubits, 3)})
        self.post = nn.Linear(n_qubits, out_features)

    def forward(self, x):
        x = self.pre(x)
        x = x.view(-1, 2, n_qubits)
        q_out = torch.stack([self.qnode(x[:,0]), self.qnode(x[:,1])], dim=1).mean(1)
        return self.post(q_out)

# =============================== Q-PGA MODEL ===============================
class Q_PGA_Unfold(nn.Module):
    def __init__(self, n_iter_outer=20, n_iter_inner=20, K=4, Nt=64, Nrf=4, M=4):
        super().__init__()
        self.K, self.Nt, self.Nrf, self.M = K, Nt, Nrf, M
        self.n_iter_inner = n_iter_inner

        # Learnable classical steps (fallback)
        self.step_size = nn.Parameter(torch.ones(n_iter_outer, n_iter_inner + K) * 0.02)

        # Quantum correctors: one per inner iteration for F
        self.q_F_updates = nn.ModuleList([
            QuantumBlock(in_features=80, out_features=Nt*Nrf*2) for _ in range(n_iter_inner)
        ])
        # One big quantum corrector for W (per outer iteration)
        self.q_W_update = QuantumBlock(in_features=80, out_features=K*Nrf*M*2)

        # State projector (giảm chiều input cho QNN)
        self.projector = nn.Linear(500, 80)  # tùy kích thước state bạn có

    def _get_state(self, H, F, W, R):
        s1 = F.abs().mean(dim=(0,1))
        s2 = F.angle().mean(dim=(0,1))
        s3 = W.abs().mean(dim=(0,1,2))
        s4 = W.angle().mean(dim=(0,1,2))
        s5 = H.abs().mean(dim=(0,1,2))
        s6 = R.abs().mean(dim=(0,1))
        state = torch.cat([s1.flatten(), s2.flatten(), s3, s4, s5.flatten(), s6.flatten()], dim=0)
        return self.projector(state.unsqueeze(0)).expand(H.shape[1], -1)

    def _to_complex(self, x, shape):
        return torch.complex(x[...,0::2], x[...,1::2]).view(*shape)

    def execute_PGA(self, H, R, Pt, n_iter_outer, n_iter_inner=None):
        n_iter_inner = n_iter_inner or self.n_iter_inner
        rate_init, tau_init, F, W = initialize(H, R, Pt, initial_normalization)

        rates = [get_sum_rate(H, F, W, Pt)]
        taus = [get_beam_error(H, F, W, R, Pt)]

        for ii in range(n_iter_outer):
            state = self._get_state(H, F, W, R)  # [batch, 80]

            for jj in range(n_iter_inner):
                # Classical gradients
                grad_F_com = get_grad_F_com(H, F, W)
                grad_F_rad = get_grad_F_rad(F, W, R)

                # Quantum correction
                q_out_F = self.q_F_updates[jj](state)
                delta_F_q = self._to_complex(q_out_F, F.shape) * 5e-4

                step_f = self.step_size[ii, 0]
                F = F + step_f * (grad_F_com - OMEGA * grad_F_rad) + delta_F_q
                F = F / (torch.abs(F) + 1e-12)

                # W update chỉ 1 lần mỗi outer iteration (như J20 gốc)
                if jj == 0:
                    grad_W_com = get_grad_W_com(H, F, W)
                    grad_W_rad = get_grad_W_rad(F, W, R)

                    q_out_W = self.q_W_update(state)
                    delta_W_q = self._to_complex(q_out_W, W.shape) * 5e-4

                    step_w = self.step_size[ii, 1:1+self.K].mean()
                    W = W + step_w * (grad_W_com - OMEGA * grad_W_rad) + delta_W_q
                    F, W = normalize(F, W, H, Pt)

            rates.append(get_sum_rate(H, F, W, Pt))
            taus.append(get_beam_error(H, F, W, R, Pt))

        rates = torch.stack(rates, dim=0).T
        taus = torch.stack(taus, dim=0).T
        return rates, taus, F, W
