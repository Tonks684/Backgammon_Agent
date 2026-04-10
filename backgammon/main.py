"""
CLI entry point for the backgammon RL agent.

Subcommands
-----------
train
    python backgammon/main.py train [--episodes N] [--eval-every N]
                                    [--config path] [--resume checkpoint.pt]

eval
    python backgammon/main.py eval --checkpoint data/checkpoints/latest.pt
                                   [--skill expert] [--matches 100]
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backgammon RL agent — train or evaluate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # train subcommand
    # ------------------------------------------------------------------
    train_p = sub.add_parser("train", help="Run self-play TD(λ) training.")
    train_p.add_argument(
        "--episodes", "-n",
        type=int,
        default=None,
        metavar="N",
        help="Total number of self-play episodes.  Overrides config value.",
    )
    train_p.add_argument(
        "--eval-every",
        type=int,
        default=None,
        metavar="N",
        help="Run gnubg evaluation every N episodes.  Overrides config value.",
    )
    train_p.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        metavar="N",
        help="Save a checkpoint every N episodes.  Overrides config value.",
    )
    train_p.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a JSON config file (see Config.from_json).  "
             "If omitted, default hyperparameters are used.",
    )
    train_p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CHECKPOINT",
        help="Resume from a saved checkpoint (.pt file).  "
             "The network weights are loaded; training continues from episode 1.",
    )
    train_p.add_argument(
        "--gnubg-eval",
        action="store_true",
        default=False,
        help="Enable periodic gnubg evaluation during training "
             "(requires gnubg on PATH).",
    )
    train_p.add_argument(
        "--skill",
        type=str,
        default="expert",
        choices=["beginner", "intermediate", "advanced", "expert", "world_class", "grandmaster"],
        help="gnubg skill level for periodic evaluation (only used with --gnubg-eval).",
    )

    # ------------------------------------------------------------------
    # eval subcommand
    # ------------------------------------------------------------------
    eval_p = sub.add_parser("eval", help="Evaluate a checkpoint against gnubg.")
    eval_p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to a saved checkpoint (.pt file).",
    )
    eval_p.add_argument(
        "--skill",
        type=str,
        default="expert",
        choices=["beginner", "intermediate", "advanced", "expert", "world_class", "grandmaster"],
        help="gnubg skill level to evaluate against.",
    )
    eval_p.add_argument(
        "--matches",
        type=int,
        default=100,
        metavar="N",
        help="Number of matches to play against gnubg.",
    )
    eval_p.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a JSON config file.  Used for network architecture params.",
    )

    return parser


def _load_config(path: str | None):
    from backgammon.config import Config
    if path is not None:
        return Config.from_json(path)
    return Config()


def _build_agent(config, checkpoint: str | None = None):
    """Instantiate ValueNetwork + TDLambdaAgent, optionally loading weights."""
    import torch
    from backgammon.agents.td_lambda import TDLambdaAgent
    from backgammon.models.mlp import ValueNetwork

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    if checkpoint is not None:
        print(f"Loading weights from {checkpoint} ...")
        network = ValueNetwork.load_checkpoint(checkpoint)
    else:
        network = ValueNetwork(
            hidden_size=config.hidden_size,
            n_hidden_layers=config.n_hidden_layers,
        )

    return TDLambdaAgent(network=network, config=config, device=device)


def cmd_train(args: argparse.Namespace) -> None:
    from backgammon.training.trainer import Trainer

    config = _load_config(args.config)
    agent = _build_agent(config, checkpoint=args.resume)

    evaluator = None
    if args.gnubg_eval:
        from backgammon.evaluation.gnubg_eval import GnubgEvaluator
        # Inject skill + matches into config for Trainer to pick up
        config.gnubg_skill = args.skill  # type: ignore[attr-defined]
        try:
            evaluator = GnubgEvaluator()
        except RuntimeError as e:
            print(f"Warning: could not initialise gnubg evaluator — {e}", file=sys.stderr)
            print("Continuing without gnubg evaluation.", file=sys.stderr)

    trainer = Trainer(agent=agent, config=config, evaluator=evaluator)
    trainer.train(
        n_episodes=args.episodes,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
    )


def cmd_eval(args: argparse.Namespace) -> None:
    from backgammon.evaluation.gnubg_eval import GnubgEvaluator

    config = _load_config(args.config)
    agent = _build_agent(config, checkpoint=args.checkpoint)

    evaluator = GnubgEvaluator()
    print(f"Evaluating {args.checkpoint} against gnubg ({args.skill}, {args.matches} matches) …")
    results = evaluator.evaluate_match(agent, skill_level=args.skill, n_matches=args.matches)

    print("\nResults:")
    print(f"  Win rate         : {results['win_rate']:.3f}")
    print(f"  Gammon rate      : {results['gammon_rate']:.3f}")
    print(f"  Backgammon rate  : {results['backgammon_rate']:.3f}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
