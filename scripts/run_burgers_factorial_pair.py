from __future__ import annotations

import argparse
from pathlib import Path

from .run_burgers_phase_seed_sensitivity import run_variant


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a matched classical-local-trigonometric and frozen-ring "
            "quantum Burgers pair with independently specified seeds."
        )
    )
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--phase-seed", type=int, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    results = []

    for variant in (
        "classical_trig_local",
        "quantum_ring_frozen",
    ):
        print("\n" + "=" * 72)
        print(
            f"{variant} | data={arguments.data_seed} | "
            f"model={arguments.model_seed} | phase={arguments.phase_seed}"
        )
        print("=" * 72)

        result = run_variant(
            variant=variant,
            data_seed=arguments.data_seed,
            model_seed=arguments.model_seed,
            phase_seed=arguments.phase_seed,
            output_root=arguments.output_root,
        )
        results.append(result)

        print(
            f"Completed {variant}: relative L2={result.relative_l2:.6f}, "
            f"time={result.training_seconds:.2f}s"
        )

    difference = (
        results[1].relative_l2
        - results[0].relative_l2
    )

    print("\nMatched pair completed")
    print(
        "quantum_minus_classical="
        f"{difference:.6f}"
    )
