from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """TD-Gammon-style MLP value network.

    Input:  54-dimensional board state vector (float32, [0, 1])
    Output: 4-dimensional probability vector
            [p_white_win, p_white_gammon, p_black_win, p_black_gammon]

    Sigmoid activations throughout — keeps all outputs as probabilities,
    consistent with the TD-Gammon convention.

    Architecture (n_hidden_layers=2, hidden_size=128):
        Linear(54 → 128) → Sigmoid → Linear(128 → 128) → Sigmoid → Linear(128 → 4) → Sigmoid
    """

    INPUT_SIZE = 54
    OUTPUT_SIZE = 4

    def __init__(self, hidden_size: int = 128, n_hidden_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_size = self.INPUT_SIZE
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.Sigmoid())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, self.OUTPUT_SIZE))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def equity(output: torch.Tensor) -> torch.Tensor:
        """Scalar equity from a 4-element output vector.

        equity = p_white_win + 2*p_white_gammon - p_black_win - 2*p_black_gammon

        Ranges from -3 (worst for white) to +3 (best for white).
        Accepts a 1-D tensor of shape (4,) or a batch of shape (N, 4).
        Returns a scalar tensor or a tensor of shape (N,).
        """
        if output.dim() == 1:
            p_ww, p_wg, p_bw, p_bg = output[0], output[1], output[2], output[3]
            return p_ww + 2.0 * p_wg - p_bw - 2.0 * p_bg
        # batched
        return (
            output[:, 0]
            + 2.0 * output[:, 1]
            - output[:, 2]
            - 2.0 * output[:, 3]
        )

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "hidden_size": self._hidden_size(),
                "n_hidden_layers": self._n_hidden_layers(),
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> ValueNetwork:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        net = cls(
            hidden_size=checkpoint["hidden_size"],
            n_hidden_layers=checkpoint["n_hidden_layers"],
        )
        net.load_state_dict(checkpoint["state_dict"])
        return net

    # ------------------------------------------------------------------
    # Helpers for checkpoint serialisation
    # ------------------------------------------------------------------

    def _hidden_size(self) -> int:
        # First layer is Linear; its out_features is hidden_size
        return self.net[0].out_features

    def _n_hidden_layers(self) -> int:
        # Each hidden layer contributes 2 modules (Linear + Sigmoid)
        # Final layer contributes 2 modules (Linear + Sigmoid)
        # Total modules = n_hidden_layers*2 + 2
        return (len(self.net) - 2) // 2
