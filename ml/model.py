"""Lightweight ResNet-style MLP Policy-Value Network for Official 101 Okey."""

from __future__ import annotations

from pathlib import Path
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.encoding import ACTION_DIM, STATE_DIM


class ResidualBlock(nn.Module):
    """Pre/post-normalized residual MLP block with LayerNorm and GELU."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.ln1(self.fc1(x)))
        h = self.ln2(self.fc2(h))
        return F.gelu(x + h)


class PolicyValueNet(nn.Module):
    """Dual-head Policy & Value ResMLP for Official 101 Okey.

    Architecture:
      Input(STATE_DIM) -> Linear(256) -> LayerNorm -> GELU
      -> 3x ResidualBlock(256, LayerNorm, GELU)
      -> Policy Head: Linear(256, ACTION_DIM)
      -> Value Head:  Linear(256, 128) -> GELU -> Linear(128, 1) -> tanh in [-1, 1]
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 256,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.input_ln = nn.LayerNorm(hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )

        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value_fc2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning `(policy_logits [B, A], value [B])`."""
        h = F.gelu(self.input_ln(self.input_proj(x)))
        for block in self.blocks:
            h = block(h)
        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_fc2(F.gelu(self.value_fc1(h)))).squeeze(-1)
        return policy_logits, value

    @torch.inference_mode()
    def predict_single(
        self,
        state_vec: np.ndarray,
        legal_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Fast single-sample inference (<1ms on CPU) returning `(masked_probs, value)`."""
        device = next(self.parameters()).device
        x = torch.from_numpy(state_vec).unsqueeze(0).to(device)
        logits, val = self.forward(x)
        logits_1d = logits[0]

        mask_t = torch.from_numpy(legal_mask).to(device=device, dtype=torch.bool)
        masked_logits = logits_1d.masked_fill(~mask_t, -1e9)
        probs = F.softmax(masked_logits, dim=-1).cpu().numpy()
        return probs, float(val.item())

    @torch.inference_mode()
    def predict_value(self, state_vec: np.ndarray) -> float:
        """Fast single-sample value-only inference."""
        device = next(self.parameters()).device
        x = torch.from_numpy(state_vec).unsqueeze(0).to(device)
        _, val = self.forward(x)
        return float(val.item())

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "num_blocks": self.num_blocks,
                "state_dict": self.state_dict(),
            },
            p,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
    ) -> PolicyValueNet:
        ckpt = torch.load(Path(path), map_location=device, weights_only=True)
        model = cls(
            state_dim=ckpt["state_dim"],
            action_dim=ckpt["action_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_blocks=ckpt["num_blocks"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model

