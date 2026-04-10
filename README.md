# Backgammon RL Agent

[![CI](https://github.com/Tonks684/Backgammon_Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Tonks684/Backgammon_Agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)

<p align="center">
  <img src="img/banner.png" alt="Backgammon RL Agent" width="720"/>
</p>

A reinforcement learning agent for backgammon built from scratch in Python, targeting **GNU Backgammon Expert level and above**. The project implements a full TD(λ) self-play training pipeline with a PyTorch MLP value network, evaluated against GNU Backgammon at progressive skill levels.

---

## Overview

Backgammon is a stochastic two-player zero-sum game — dice rolls introduce chance nodes that distinguish it from deterministic games like Chess or Go. This project follows the two-phase approach pioneered by TD-Gammon (Tesauro, 1992):

1. **Phase 1 — TD(λ) baseline**: MLP value network trained entirely via self-play using temporal-difference learning with eligibility traces. Target: Expert / World Class level.
2. **Phase 2 — Stochastic MuZero**: MCTS with chance nodes (expectiminimax). Proven SOTA on backgammon, surpassing GNUbg Grandmaster level.

### Architecture

```
Input (54-dim board vector)
    ↓
Linear(54 → 128) → Sigmoid
    ↓
Linear(128 → 128) → Sigmoid
    ↓
Linear(128 → 4) → Sigmoid
    ↓
Output: [p_white_win, p_white_gammon, p_black_win, p_black_gammon]
```

The 54-dimensional state vector encodes checker positions, bar counts, borne-off counts, pip count, and game phase — always from the current player's perspective.

---

## Project Structure

```
reinforcement_agents/
  backgammon/
    game/
      board.py          # Board state, move generation, full rules engine
      encoder.py        # Board → 54-dim float32 vector
      types.py          # Move, Player, GameResult, DiceRoll
    agents/
      td_lambda.py      # TD(λ) agent: select_move + TD weight update
      random_agent.py   # Uniform random baseline
    training/
      self_play.py      # Self-play game loop (both sides same agent)
      trainer.py        # Orchestrates training, checkpointing, wandb logging
    evaluation/
      gnubg_eval.py     # GNU Backgammon subprocess evaluator
      metrics.py        # WinRateTracker, compute_equity_target
    models/
      mlp.py            # PyTorch MLP (ValueNetwork)
    main.py             # CLI entry point
    config.py           # Hyperparameter dataclass
  tests/                # pytest unit tests (game engine + encoder)
  img/                  # Project images
  requirements.txt
  LICENSE
```

---

## Getting Started

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt

# GNU Backgammon (for evaluation, Oracle VM / Linux only)
sudo apt-get install gnubg
```

### Training

```bash
# Train with default config (500K self-play episodes)
python backgammon/main.py train

# Custom episodes, with periodic gnubg evaluation
python backgammon/main.py train --episodes 2000000 --gnubg-eval --skill expert

# Resume from a checkpoint
python backgammon/main.py train --resume data/checkpoints/latest.pt

# Disable wandb
WANDB_MODE=disabled python backgammon/main.py train --episodes 1000
```

### Evaluation

```bash
# Evaluate a saved checkpoint against gnubg Expert (100 matches)
python backgammon/main.py eval \
    --checkpoint data/checkpoints/latest.pt \
    --skill expert \
    --matches 100
```

Available skill levels: `beginner`, `intermediate`, `advanced`, `expert`, `world_class`, `grandmaster`.

---

## Evaluation Benchmarks

| Opponent | Target win rate |
|---|---|
| Random agent | > 99% |
| gnubg Beginner | > 90% |
| gnubg Intermediate | > 75% |
| gnubg Advanced | > 60% |
| gnubg Expert | > 50% |
| gnubg World Class | > 50% |
| gnubg Grandmaster | > 50% |

Performance is also measured by **Player Rating (PR)** — average equity loss per decision × 500. World-class human players achieve PR 3–5.

---

## Training Metrics (wandb)

| Frequency | Metrics logged |
|---|---|
| Every episode | `td_loss`, `game_length` |
| Every `eval_every` episodes | `self_play_win_rate`, `gammon_rate` |
| Every gnubg eval | `gnubg_{skill}_win_rate` |

Set `WANDB_MODE=disabled` to skip wandb entirely.

---

## CI

GitHub Actions runs the full test suite (`pytest tests/`) on Python 3.10, 3.11, and 3.12 on every push and pull request to `master`.

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the workflow definition.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## References

- [TD-Gammon (Tesauro, 1995)](https://bkgm.com/articles/tesauro/tdl.html)
- [Stochastic MuZero (DeepMind/UCL, 2022)](https://openreview.net/pdf?id=X6D9bAHhBQ1)
- [GNU Backgammon](https://www.gnu.org/software/gnubg/)
- [WildBG — modern neural net engine](https://github.com/carsten-wenderdel/wildbg)
- [arXiv:2504.02221 — Learning and Improving Backgammon Strategy (2025)](https://arxiv.org/abs/2504.02221)
- [gym-backgammon](https://github.com/dellalibera/gym-backgammon)
