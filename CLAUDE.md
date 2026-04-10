# Reinforcement Learning Backgammon Agent

## Project Overview

Building a backgammon RL agent from scratch. Backgammon is a stochastic two-player zero-sum game — dice rolls introduce chance nodes that distinguish it from deterministic games like Chess or Go. The goal is to design the full stack: data sources, storage, state representation, algorithm, training infrastructure, and evaluation. Target performance: GNU Backgammon "Expert" level minimum, aiming for "World Class" or above.

## Infrastructure

- **Training compute**: Oracle VM (available for self-play and model training)
- **Development**: Windows local machine (C:\Users\Samuel Tonks\reinforcement_agents)

---

## State of the Art

### Milestone History
- **TD-Gammon (Tesauro, 1992–1995)**: First superhuman RL backgammon agent. 3-layer MLP + TD(λ) + self-play, 1.5M games, ~198 input features. The foundational reference.
- **GNU Backgammon (gnubg)**: Multi-network MLP (separate contact/crashed/race networks), TD(λ), approaches Grandmaster level. Open source, free, used as evaluation oracle.
- **Stochastic MuZero (DeepMind/UCL, 2022)**: Current SOTA. Extends MuZero with chance nodes via *afterstate dynamics*. Surpassed GNUbg Grandmaster with smaller search budgets.

### Key Insight: Stochastic Structure
Backgammon has 21 distinct dice outcomes per turn. Algorithms must handle **chance nodes**. The winning approach (Stochastic MuZero) factorises transitions as:
1. **Action → Afterstate** (deterministic): board after your move, before dice roll
2. **Afterstate → Next State** (stochastic): the random dice outcome

---

## Data Strategy

### Available Datasets
| Dataset | Size | Format | Access |
|---|---|---|---|
| Big Brother Database | ~3,000 matches from top FIBS players | .mat (Jellyfish) | BGBlitz website |
| Little Sister Archive | ~20,000 matches from FIBS | .mat | François Hochede |
| FIBS live games | Ongoing | oldmoves format | fibs.com |
| Self-play data | Unlimited | Custom numpy/HDF5 | Generated during training |

**Primary training data**: self-generated via self-play. Datasets above are optional for supervised pretraining to speed convergence.

### File Formats
- **.mat** — Jellyfish format; supported by gnubg, Snowie, BGBlitz — most real-world games
- **.sgf** — GNU Backgammon native; best for archival
- **HDF5** — for storing numpy arrays of board states during training

### Storage Architecture
```
reinforcement_agents/
  data/
    raw/            # Downloaded .mat/.sgf game logs
    processed/      # Numpy arrays, HDF5 datasets
    checkpoints/    # Model weights (.pt or JAX orbax)
    replays/        # Experience replay buffer snapshots
    evals/          # Evaluation results vs gnubg levels
```

- **< 100K games**: SQLite or flat HDF5
- **> 100K games**: PostgreSQL or sharded HDF5
- **Experience replay buffer**: In-memory ring buffer (numpy), snapshot to disk periodically

---

## State Representation

### Board Encoding (54-dimensional vector)
```python
state = [
  white_pieces[0:24],   # checker counts on each of 24 points (0–15)
  black_pieces[0:24],   # checker counts on each of 24 points (0–15)
  white_bar,            # checkers on bar
  black_bar,
  white_borne_off,      # checkers already borne off
  black_borne_off,
  pip_count_white,      # total pip distance to bear off (normalised)
  pip_count_black,
  turn,                 # 0 = white, 1 = black
  phase,                # 0 = contact, 1 = race, 2 = bearing off
]
# Shape: (54,), dtype: float32, normalised to [0, 1]
```

No complex feature engineering needed beyond pip counts and phase — these are structurally important and aid convergence. The neural network learns everything else.

---

## Algorithm Selection

### Recommended: Two-Phase Approach

#### Phase 1 — TD(λ) Baseline (implement first)
- Simplest correct algorithm for backgammon
- Well understood, easy to debug
- Reaches Expert/World Class level (~100–500K self-play games)
- **Architecture**: MLP, 2–3 hidden layers, 128–256 units each
- **Output**: 4-head (prob: white normal win, white gammon win, black normal win, black gammon win)
- **Move selection**: enumerate all legal moves after dice roll → evaluate each with network → pick max expected value
- Establishes a solid baseline before moving to more complex algorithms

#### Phase 2 — Stochastic MuZero (scale up)
- Adds MCTS with chance nodes (expectiminimax tree)
- Learns dynamics model (transition function) + value + policy heads
- Proven SOTA on backgammon; surpasses GNUbg Grandmaster
- Reference implementation: https://github.com/DHDev0/Stochastic-muzero
- Switch to JAX + Flax for vectorised self-play at this stage

**Decision point**: Implement TD(λ) first. Evaluate against gnubg. If Expert/World Class is sufficient, stop. If aiming for Grandmaster+, move to Stochastic MuZero.

---

## Training Infrastructure

### Framework
- **Phase 1**: PyTorch — accessible, fast to iterate, large ecosystem
- **Phase 2**: JAX + Flax — 1000x speedup for vectorised self-play, critical for MCTS scale-up

### Self-Play Loop (TD-λ)
```
1. Initialise neural network (random weights)
2. Roll dice → enumerate legal moves → evaluate each with NN → select best
3. Play full game (both sides use same NN)
4. At game end, compute TD(λ) targets along trajectory
5. Update NN weights via backprop
6. Periodically checkpoint + evaluate vs gnubg
7. Repeat (target: 500K–2M games)
```

### Key Libraries
| Library | Role |
|---|---|
| `torch` / `flax` | Neural network |
| `numpy` | State arrays, replay buffer |
| `gymnasium` + `gym-backgammon` | Game environment wrapper |
| `openspiel` | Backgammon game logic + MCTS infrastructure (Phase 2) |
| `h5py` | HDF5 storage |
| `wandb` | Training metrics / logging |

### Compute Estimates (Oracle VM)
- **Expert level**: ~100–500 GPU-hours
- **World Class / Grandmaster**: 500–2000 GPU-hours
- Start single-GPU; parallelise with vectorised envs when scaling up

---

## Evaluation Framework

### Progressive Benchmarks (in order)
| Benchmark | Target win rate |
|---|---|
| vs random agent | > 99% |
| vs gnubg Beginner | > 90% |
| vs gnubg Intermediate | > 75% |
| vs gnubg Advanced | > 60% |
| vs gnubg Expert | > 50% (solid baseline) |
| vs gnubg World Class | > 50% (strong result) |
| vs gnubg Grandmaster | > 50% (SOTA territory) |

### Equity Metrics
- **PR (Player Rating)** = average equity loss per decision × 500
  - PR 3–5: World-class human
  - PR 5–10: Very good amateur
  - PR 10+: Average player
- Use gnubg's analysis tools to compute PR against our agent's decisions

### FIBS (stretch goal)
- Create a bot account on FIBS (fibs.com)
- Elo rating against thousands of human players
- Rating > 1800 = very strong; > 2000 = world-class territory

---

## Implementation Phases

### Phase 0: Environment Setup
- [ ] Implement backgammon game engine (or wrap gym-backgammon)
  - Full rules: movement, hitting, bearing off, win/gammon/backgammon detection
  - Legal move generation for all 21 dice outcomes
- [ ] Implement state encoder (54-dim vector)
- [ ] Write unit tests for game logic
- [ ] Set up gnubg as evaluation oracle (install on Oracle VM, Python subprocess interface)

### Phase 1: TD(λ) Baseline
- [ ] Implement MLP value network in PyTorch (54 → 128 → 128 → 4)
- [ ] Implement TD(λ) training loop with eligibility traces
- [ ] Implement self-play loop
- [ ] Add wandb logging (loss, win rates, PR)
- [ ] Run 500K self-play games on Oracle VM
- [ ] Evaluate against gnubg skill levels

### Phase 2: MCTS Enhancement
- [ ] Integrate OpenSpiel or implement expectiminimax with chance nodes
- [ ] Add policy head to network (alongside value head)
- [ ] Implement Stochastic MuZero (or simplified 2-ply expectiminimax + NN)
- [ ] Switch to JAX for vectorised self-play speedup
- [ ] Run 2M+ self-play games on Oracle VM
- [ ] Evaluate for Grandmaster-level performance

### Phase 3: Pretraining & Fine-tuning (Optional)
- [ ] Download Big Brother / Little Sister datasets
- [ ] Parse .mat files → board states + moves
- [ ] Supervised pretraining to initialise network
- [ ] Fine-tune with self-play RL

---

## Target File Structure

```
reinforcement_agents/
  backgammon/
    game/
      board.py          # Board state, move generation, rules
      encoder.py        # State → 54-dim numpy vector
      types.py          # Move, State, Player datatypes
    agents/
      td_lambda.py      # TD(λ) training + value network
      mcts.py           # Expectiminimax / Stochastic MuZero
      random_agent.py   # Baseline
    training/
      self_play.py      # Self-play loop
      replay_buffer.py  # Experience storage
      trainer.py        # Orchestrates training
    evaluation/
      gnubg_eval.py     # Interface with gnubg subprocess
      metrics.py        # Win rate, PR calculation
    data/
      parser.py         # .mat/.sgf file parsing
      storage.py        # HDF5 read/write
    models/
      mlp.py            # PyTorch MLP value/policy network
    main.py             # Entry point
    config.py           # Hyperparameters
  CLAUDE.md
```

---

## Key References

- [TD-Gammon (Tesauro, 1995)](https://bkgm.com/articles/tesauro/tdl.html)
- [Stochastic MuZero paper](https://openreview.net/pdf?id=X6D9bAHhBQ1)
- [gym-backgammon](https://github.com/dellalibera/gym-backgammon)
- [OpenSpiel (DeepMind)](https://github.com/google-deepmind/open_spiel)
- [Stochastic MuZero PyTorch impl](https://github.com/DHDev0/Stochastic-muzero)
- [WildBG — modern neural net engine](https://github.com/carsten-wenderdel/wildbg)
- [arXiv:2504.02221 — Learning and Improving Backgammon Strategy (2025)](https://arxiv.org/abs/2504.02221)
- [GNU Backgammon](https://www.gnu.org/software/gnubg/)
- [FIBS](http://www.fibs.com)

---

## Oracle VM Setup

### Initial setup (run once)

```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y gnubg python3-pip python3-venv

# Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # installs the backgammon package in editable mode
```

### Training

```bash
# Default config (500K episodes, eval vs gnubg every 10K)
python backgammon/main.py train

# Custom episode count with gnubg evaluation enabled
python backgammon/main.py train --episodes 2000000 --gnubg-eval --skill expert

# Resume from a checkpoint
python backgammon/main.py train --resume data/checkpoints/latest.pt

# Disable wandb (e.g. for quick smoke-test)
WANDB_MODE=disabled python backgammon/main.py train --episodes 1000
```

### Evaluation

```bash
# Evaluate a saved checkpoint against gnubg expert (100 matches)
python backgammon/main.py eval --checkpoint data/checkpoints/latest.pt --skill expert

# More matches for a tighter estimate
python backgammon/main.py eval --checkpoint data/checkpoints/latest.pt \
    --skill world_class --matches 500
```

### wandb

Training metrics are logged automatically if `wandb` is installed and
`WANDB_MODE` is not set to `disabled`.  Per-episode: `td_loss`, `game_length`.
Per `eval_every`: `self_play_win_rate`, `gammon_rate`.
Per gnubg eval: `gnubg_{skill}_win_rate`.
