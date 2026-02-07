#!/usr/bin/env python3
"""Run DSPy optimization (MIPROv2 or BootstrapFewShot) on the library card extraction task."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import configure_dspy, resolve_model, RESULTS_DIR
from src.data import load_and_split
from src.module import LibraryCardExtractor
from src.scoring import dspy_metric, gepa_feedback_metric

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_mipro(trainset, devset, auto="light", max_bootstrapped=2, max_labeled=2, num_threads=8, module_type="predict"):
    """Run MIPROv2 optimization."""
    import dspy

    optimizer = dspy.MIPROv2(
        metric=dspy_metric,
        auto=auto,
        num_threads=num_threads,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_labeled,
    )
    extractor = LibraryCardExtractor(module_type=module_type)
    optimized = optimizer.compile(
        extractor,
        trainset=trainset,
        valset=devset,
    )
    return optimized


def run_bootstrap(trainset, max_bootstrapped=3, max_labeled=3, num_threads=8, module_type="predict"):
    """Run BootstrapFewShot optimization."""
    import dspy

    optimizer = dspy.BootstrapFewShot(
        metric=dspy_metric,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_labeled,
        num_threads=num_threads,
    )
    extractor = LibraryCardExtractor(module_type=module_type)
    optimized = optimizer.compile(extractor, trainset=trainset)
    return optimized


def run_simba(trainset, module_type="cot", num_threads=8, max_steps=8, num_candidates=6, bsize=32):
    """Run SIMBA optimization."""
    from dspy.teleprompt import SIMBA

    optimizer = SIMBA(
        metric=dspy_metric,
        bsize=min(bsize, len(trainset)),
        num_candidates=num_candidates,
        max_steps=max_steps,
        max_demos=4,
        num_threads=num_threads,
    )
    extractor = LibraryCardExtractor(module_type=module_type)
    return optimizer.compile(extractor, trainset=trainset, seed=42)


def run_gepa(trainset, devset, auto="light", module_type="cot", reflection_model=None, num_threads=8):
    """Run GEPA optimization."""
    import dspy
    from dspy.teleprompt import GEPA
    from dspy.teleprompt.gepa.instruction_proposal import MultiModalInstructionProposer

    reflection_lm = dspy.LM(resolve_model(reflection_model), temperature=1.0, max_tokens=16000)
    optimizer = GEPA(
        metric=gepa_feedback_metric,
        auto=auto,
        reflection_lm=reflection_lm,
        instruction_proposer=MultiModalInstructionProposer(),
        num_threads=num_threads,
    )
    extractor = LibraryCardExtractor(module_type=module_type)
    return optimizer.compile(extractor, trainset=trainset, valset=devset)


def main():
    parser = argparse.ArgumentParser(description="Optimize library card extraction with DSPy")
    parser.add_argument(
        "--optimizer",
        choices=["mipro", "bootstrap", "simba", "gepa"],
        default="mipro",
        help="Optimizer to use (default: mipro)",
    )
    parser.add_argument("--auto", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--max-bootstrapped", type=int, default=2)
    parser.add_argument("--max-labeled", type=int, default=2)
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model preset or full model string")
    parser.add_argument("--num-threads", type=int, default=8, help="Number of parallel threads for evaluation")
    parser.add_argument("--module", choices=["predict", "cot"], default="predict", help="Module type: predict or cot (ChainOfThought)")
    parser.add_argument("--seed", type=int, default=42)
    # SIMBA-specific
    parser.add_argument("--max-steps", type=int, default=8, help="SIMBA: optimization steps (default: 8)")
    parser.add_argument("--num-candidates", type=int, default=6, help="SIMBA: candidates per step (default: 6)")
    parser.add_argument("--bsize", type=int, default=32, help="SIMBA: mini-batch size (default: 32)")
    # GEPA-specific
    parser.add_argument("--reflection-model", type=str, default=None, help="GEPA: model for reflection LM (e.g. gemini-2.5-pro)")
    args = parser.parse_args()

    model_id = resolve_model(args.model)
    logger.info(f"Using model: {model_id}")
    configure_dspy(model=args.model)

    train_ex, dev_ex, test_ex, *_ = load_and_split(seed=args.seed)
    logger.info(f"Data split: train={len(train_ex)}, dev={len(dev_ex)}, test={len(test_ex)}")

    if args.optimizer == "mipro":
        logger.info(f"Running MIPROv2 (auto={args.auto}, module={args.module}, bootstrapped={args.max_bootstrapped}, labeled={args.max_labeled}, threads={args.num_threads})")
        optimized = run_mipro(
            train_ex, dev_ex,
            auto=args.auto,
            max_bootstrapped=args.max_bootstrapped,
            max_labeled=args.max_labeled,
            num_threads=args.num_threads,
            module_type=args.module,
        )
    elif args.optimizer == "bootstrap":
        logger.info(f"Running BootstrapFewShot (module={args.module}, bootstrapped={args.max_bootstrapped}, labeled={args.max_labeled}, threads={args.num_threads})")
        optimized = run_bootstrap(
            train_ex,
            max_bootstrapped=args.max_bootstrapped,
            max_labeled=args.max_labeled,
            num_threads=args.num_threads,
            module_type=args.module,
        )
    elif args.optimizer == "simba":
        logger.info(f"Running SIMBA (module={args.module}, steps={args.max_steps}, candidates={args.num_candidates}, bsize={args.bsize}, threads={args.num_threads})")
        optimized = run_simba(
            train_ex,
            module_type=args.module,
            num_threads=args.num_threads,
            max_steps=args.max_steps,
            num_candidates=args.num_candidates,
            bsize=args.bsize,
        )
    elif args.optimizer == "gepa":
        if args.reflection_model is None:
            parser.error("--reflection-model is required when --optimizer=gepa")
        logger.info(f"Running GEPA (auto={args.auto}, module={args.module}, reflection={args.reflection_model}, threads={args.num_threads})")
        optimized = run_gepa(
            train_ex, dev_ex,
            auto=args.auto,
            module_type=args.module,
            reflection_model=args.reflection_model,
            num_threads=args.num_threads,
        )

    # Save optimized program
    out_dir = RESULTS_DIR / "optimized"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.replace("/", "_")
    module_tag = f"-{args.module}" if args.module != "predict" else ""
    save_path = out_dir / f"{args.optimizer}{module_tag}_{model_tag}_optimized.json"
    optimized.save(str(save_path))
    logger.info(f"Optimized program saved to {save_path}")


if __name__ == "__main__":
    main()
