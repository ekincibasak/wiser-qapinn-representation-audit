"""Generate the bundled Cole-Hopf / Gauss-Hermite Burgers reference grid."""
from pathlib import Path

from burgers import save_burgers_reference, solve_burgers_reference


def main() -> None:
    output = Path("data/burgers_reference.npz")
    reference = solve_burgers_reference(
        viscosity=0.01 / 3.141592653589793,
        spatial_points=401,
        quadrature_order=120,
    )
    save_burgers_reference(reference, output)
    print(f"Saved {output} with shape {reference.u.shape}")


if __name__ == "__main__":
    main()
