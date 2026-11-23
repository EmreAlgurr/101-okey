"""Supervised warm-start data generation and training for PolicyValueNet.

Generates headless 4-player games using `HeuristicAgent`, records every
non-trivial `(state_vec, legal_mask, chosen_action_idx, normalized_final_score)`,
and trains `PolicyValueNet` with masked Cross-Entropy + Value MSE on CUDA/CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from agents.heuristic_agent import HeuristicAgent
from engine import GameConfig, NUM_PLAYERS, Okey101Engine
from ml.encoding import (
    ACTION_DIM,
    STATE_DIM,
    action_to_index,
    encode_obs,
    legal_action_mask,
    normalize_score,
)
from ml.model import PolicyValueNet


def generate_dataset(
    num_games: int,
    seed: int = 42,
    katlamali: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Play `num_games` headless 4x Heuristic games and collect training samples."""
    states_list: list[np.ndarray] = []
    masks_list: list[np.ndarray] = []
    actions_list: list[int] = []
    values_list: list[float] = []

    config = GameConfig(katlamali=katlamali)
    t0 = time.perf_counter()

    for g in range(num_games):
        engine = Okey101Engine(config=config)
        engine.reset(seed=seed + g, starting_player=g % NUM_PLAYERS)
        agents = [HeuristicAgent(seed=seed + g * 13 + p) for p in range(NUM_PLAYERS)]

        round_records: list[tuple[np.ndarray, np.ndarray, int, int]] = []

        while not engine.done:
            p = engine.current_player
            obs = engine.get_observation(p)
            legal = engine.get_legal_actions()
            chosen = agents[p].select_action(obs, legal)

            # Record all non-trivial decisions (len(legal) > 1)
            if len(legal) > 1:
                s_vec = encode_obs(obs)
                mask = legal_action_mask(obs, legal)
                a_idx = action_to_index(chosen, obs)
                round_records.append((s_vec, mask, a_idx, p))

            engine.step(chosen)

        assert engine.round_scores is not None
        norm_scores = [normalize_score(s) for s in engine.round_scores]
        for s_vec, mask, a_idx, p in round_records:
            states_list.append(s_vec)
            masks_list.append(mask)
            actions_list.append(a_idx)
            values_list.append(norm_scores[p])

        if (g + 1) % max(1, num_games // 5) == 0 or g + 1 == num_games:
            dt = time.perf_counter() - t0
            print(
                f"  Generated {g + 1}/{num_games} games "
                f"({len(states_list):,} samples) in {dt:.1f}s",
                flush=True,
            )

    return (
        np.asarray(states_list, dtype=np.float32),
        np.asarray(masks_list, dtype=np.bool_),
        np.asarray(actions_list, dtype=np.int64),
        np.asarray(values_list, dtype=np.float32),
    )


def train_supervised(
    num_games: int = 2000,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    checkpoint_path: str = "checkpoints/pv_net_v1.pt",
    seed: int = 42,
    katlamali: bool = False,
) -> PolicyValueNet:
    """Generate heuristic games and train PolicyValueNet with supervised warm-start."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training PolicyValueNet on device: {device}", flush=True)
    print(f"Generating {num_games} headless 4x Heuristic games...", flush=True)

    states, masks, actions, values = generate_dataset(
        num_games=num_games, seed=seed, katlamali=katlamali
    )
    n_samples = len(states)
    perm = np.random.permutation(n_samples)
    states, masks, actions, values = (
        states[perm],
        masks[perm],
        actions[perm],
        values[perm],
    )

    val_size = max(1, int(0.1 * n_samples))
    train_ds = TensorDataset(
        torch.from_numpy(states[val_size:]),
        torch.from_numpy(masks[val_size:]),
        torch.from_numpy(actions[val_size:]),
        torch.from_numpy(values[val_size:]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(states[:val_size]),
        torch.from_numpy(masks[:val_size]),
        torch.from_numpy(actions[:val_size]),
        torch.from_numpy(values[:val_size]),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PolicyValueNet(state_dim=STATE_DIM, action_dim=ACTION_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    ckpt_file = Path(checkpoint_path)

    print(
        f"\nDataset: {len(train_ds):,} train / {len(val_ds):,} val samples | "
        f"STATE_DIM={STATE_DIM}, ACTION_DIM={ACTION_DIM}",
        flush=True,
    )
    print("-" * 82, flush=True)
    print(
        f"{'Epoch':>5} | {'Tr PolAcc':>9} | {'Tr ValMSE':>9} | "
        f"{'Va PolAcc':>9} | {'Va ValMSE':>9} | {'Va Loss':>8}",
        flush=True,
    )
    print("-" * 82, flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        tr_correct = 0
        tr_total = 0
        tr_val_sq_err = 0.0

        for s_b, m_b, a_b, v_b in train_loader:
            s_b = s_b.to(device)
            m_b = m_b.to(device)
            a_b = a_b.to(device)
            v_b = v_b.to(device)

            logits, pred_v = model(s_b)
            masked_logits = logits.masked_fill(~m_b, -1e9)

            pol_loss = F.cross_entropy(masked_logits, a_b)
            val_loss = F.mse_loss(pred_v, v_b)
            loss = pol_loss + 2.0 * val_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = masked_logits.argmax(dim=-1)
            tr_correct += int((preds == a_b).sum().item())
            tr_total += len(a_b)
            tr_val_sq_err += float(((pred_v - v_b) ** 2).sum().item())

        scheduler.step()

        # Validation pass
        model.eval()
        va_correct = 0
        va_total = 0
        va_val_sq_err = 0.0
        va_loss_sum = 0.0

        with torch.inference_mode():
            for s_b, m_b, a_b, v_b in val_loader:
                s_b = s_b.to(device)
                m_b = m_b.to(device)
                a_b = a_b.to(device)
                v_b = v_b.to(device)

                logits, pred_v = model(s_b)
                masked_logits = logits.masked_fill(~m_b, -1e9)
                pol_loss = F.cross_entropy(masked_logits, a_b)
                val_loss = F.mse_loss(pred_v, v_b)

                va_loss_sum += float((pol_loss + 2.0 * val_loss).item()) * len(a_b)
                preds = masked_logits.argmax(dim=-1)
                va_correct += int((preds == a_b).sum().item())
                va_total += len(a_b)
                va_val_sq_err += float(((pred_v - v_b) ** 2).sum().item())

        tr_acc = 100.0 * tr_correct / max(1, tr_total)
        tr_mse = tr_val_sq_err / max(1, tr_total)
        va_acc = 100.0 * va_correct / max(1, va_total)
        va_mse = va_val_sq_err / max(1, va_total)
        va_loss = va_loss_sum / max(1, va_total)

        print(
            f"{epoch:>5d} | {tr_acc:>8.2f}% | {tr_mse:>9.4f} | "
            f"{va_acc:>8.2f}% | {va_mse:>9.4f} | {va_loss:>8.4f}",
            flush=True,
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            model.save_checkpoint(ckpt_file)

    print("-" * 82, flush=True)
    print(f"Saved best checkpoint to {ckpt_file} (val_loss={best_val_loss:.4f})", flush=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supervised warm-start training for 101 Okey PolicyValueNet."
    )
    parser.add_argument("--games", type=int, default=2000, help="Number of heuristic games.")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--out", type=str, default="checkpoints/pv_net_v1.pt", help="Output checkpoint path."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--katlamali", action="store_true", help="Use katlamali rules.")
    args = parser.parse_args()

    train_supervised(
        num_games=args.games,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_path=args.out,
        seed=args.seed,
        katlamali=args.katlamali,
    )


if __name__ == "__main__":
    main()

