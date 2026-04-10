# Backgammon TD(λ) — Hyperparameter Optimisation

## Objective

Maximise win rate against a random opponent after a 5-minute training run
on a single GPU. The score is printed as `score: X.XXXX` at the end of
`train.py` — higher is better (range 0.0–1.0, random baseline ≈ 0.50).

## Current best known score

score: (update this after each experiment)

## What you MAY change in train.py

Everything below the `# TUNABLE PARAMETERS` comment:
- `ALPHA` — TD learning rate (try: 0.001, 0.005, 0.01, 0.05)
- `LAMBDA` — eligibility trace decay (try: 0.5, 0.7, 0.8, 0.9)
- `HIDDEN_SIZE` — neurons per hidden layer (try: 64, 128, 256)
- `N_HIDDEN_LAYERS` — depth of MLP (try: 1, 2, 3)
- `BATCH_SIZE` — games per update step (try: 16, 32, 64)
- `N_WORKERS` — parallel game workers (keep <= CPU core count, max 32)
- Network activation functions in ValueNetwork (currently Sigmoid throughout)
- The TD update formulation in TDLambdaAgent.update()
- Learning rate schedules (e.g. decay alpha over time)

## What you MUST NOT change

- `BUDGET_SECONDS = 300` (fixed experiment window)
- `evaluate_vs_random()` function (evaluation must stay consistent)
- Anything in `backgammon/game/` (board rules, encoder, types are fixed)
- The final `score: X.XXXX` print format (autoresearch reads this line)

## Known issues to investigate

- Loss was rising after 500 steps with ALPHA=0.1 — likely too high
- Current ALPHA=0.01 is a conservative starting point, may be too low
- Eligibility traces may be accumulating too aggressively with long games

## Constraints

- Lightning.ai free plan: 32 CPU cores, single GPU (L40S/A10/H100/H200)
- 50GB persistent storage — checkpoints are small, not a concern
- 5-minute experiment window — optimise for fast convergence signal

## Strategy hints

Start by tuning ALPHA and LAMBDA (highest leverage).
Then explore HIDDEN_SIZE once learning dynamics are stable.
Avoid very large N_HIDDEN_LAYERS — backgammon is a relatively simple
function approximation task and deep networks may overfit or diverge.
