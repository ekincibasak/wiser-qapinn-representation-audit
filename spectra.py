from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class FrequencyBands:
    """Frequency-band limits expressed as fractions of the Nyquist frequency.

    The default definitions are:

        low:    0%  <= f/f_Nyquist < 20%
        middle: 20% <= f/f_Nyquist < 50%
        high:   50% <= f/f_Nyquist <= 100%

    These limits are analysis choices, not universal physical constants.
    They must be kept identical when comparing classical and quantum models.
    """

    low_upper_fraction: float = 0.20
    middle_upper_fraction: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 < self.low_upper_fraction < 1.0:
            raise ValueError(
                "low_upper_fraction must lie strictly between 0 and 1."
            )
        if not (
            self.low_upper_fraction
            < self.middle_upper_fraction
            < 1.0
        ):
            raise ValueError(
                "middle_upper_fraction must be larger than "
                "low_upper_fraction and smaller than 1."
            )


@dataclass(frozen=True)
class BandMetrics:
    """Spectral metrics for one frequency band."""

    frequency_min: float
    frequency_max: float
    number_of_bins: int

    reference_energy: float
    prediction_energy: float
    error_energy: float

    reference_energy_fraction: float
    prediction_energy_fraction: float

    error_normalized_by_total_reference_energy: float
    relative_band_error: float | None


@dataclass(frozen=True)
class SpectrumResult:
    """One-sided Fourier spectrum and metadata."""

    frequencies: np.ndarray
    coefficients: np.ndarray
    amplitudes: np.ndarray
    nyquist_frequency: float
    spatial_step: float
    used_odd_extension: bool


def validate_spatial_data(
    x: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate a one-dimensional field on a uniform spatial grid."""
    x = np.asarray(x, dtype=float)
    values = np.asarray(values, dtype=float)

    if x.ndim != 1:
        raise ValueError(f"x must be one-dimensional, got shape {x.shape}.")
    if values.ndim != 1:
        raise ValueError(
            f"values must be one-dimensional, got shape {values.shape}."
        )
    if x.size != values.size:
        raise ValueError(
            "x and values must contain the same number of samples."
        )
    if x.size < 4:
        raise ValueError("At least four spatial samples are required.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(values)):
        raise ValueError("x and values must contain only finite numbers.")

    differences = np.diff(x)

    if not np.all(differences > 0.0):
        raise ValueError("x must be strictly increasing.")

    dx = float(np.mean(differences))

    if not np.allclose(
        differences,
        dx,
        rtol=1.0e-5,
        atol=1.0e-8,
    ):
        raise ValueError(
            "The Fourier analysis requires a uniformly spaced x grid."
        )

    return x, values, dx


def odd_dirichlet_extension(values: np.ndarray) -> np.ndarray:
    """Create an odd periodic extension for zero-Dirichlet boundaries.

    For samples on [0, L] including both endpoints, this constructs samples
    over one period [0, 2L):

        [u(0), ..., u(L), -u(L-dx), ..., -u(dx)].

    A mode sin(n*pi*x/L) then aligns with a Fourier frequency n/(2L).
    """
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")

    return np.concatenate(
        [
            values,
            -values[-2:0:-1],
        ]
    )


def compute_spatial_fourier_spectrum(
    x: np.ndarray,
    values: np.ndarray,
    *,
    use_odd_extension: bool = True,
) -> SpectrumResult:
    """Compute a one-sided orthonormal spatial Fourier spectrum.

    Parameters
    ----------
    x:
        Uniform spatial coordinates.
    values:
        One spatial solution profile u(x, t_fixed).
    use_odd_extension:
        When True, use an odd extension suited to zero-Dirichlet boundaries.
        When False, apply the FFT directly to samples on [0, L].

    Notes
    -----
    ``norm="ortho"`` gives a unitary discrete Fourier transform. This makes
    squared coefficient magnitudes suitable for consistent energy comparisons.
    """
    _, values, dx = validate_spatial_data(x=x, values=values)

    transformed_values = (
        odd_dirichlet_extension(values)
        if use_odd_extension
        else values
    )

    coefficients = np.fft.rfft(
        transformed_values,
        norm="ortho",
    )
    frequencies = np.fft.rfftfreq(
        transformed_values.size,
        d=dx,
    )
    amplitudes = np.abs(coefficients)

    return SpectrumResult(
        frequencies=frequencies,
        coefficients=coefficients,
        amplitudes=amplitudes,
        nyquist_frequency=float(frequencies[-1]),
        spatial_step=dx,
        used_odd_extension=use_odd_extension,
    )


def construct_band_masks(
    frequencies: np.ndarray,
    bands: FrequencyBands,
) -> dict[str, np.ndarray]:
    """Construct low-, middle-, and high-frequency masks."""
    frequencies = np.asarray(frequencies, dtype=float)

    if frequencies.ndim != 1 or frequencies.size < 2:
        raise ValueError("frequencies must be a one-dimensional array.")

    nyquist = float(frequencies[-1])

    if nyquist <= 0.0:
        raise ValueError("Nyquist frequency must be positive.")

    normalized_frequency = frequencies / nyquist

    return {
        "low": normalized_frequency < bands.low_upper_fraction,
        "middle": (
            (normalized_frequency >= bands.low_upper_fraction)
            & (
                normalized_frequency
                < bands.middle_upper_fraction
            )
        ),
        "high": (
            normalized_frequency
            >= bands.middle_upper_fraction
        ),
    }


def calculate_band_metrics(
    prediction_spectrum: SpectrumResult,
    reference_spectrum: SpectrumResult,
    bands: FrequencyBands,
    *,
    minimum_reference_energy_fraction: float = 1.0e-12,
) -> dict[str, BandMetrics]:
    """Calculate stable spectral errors for each frequency band.

    Two error normalizations are reported:

    1. ``error_normalized_by_total_reference_energy`` is always meaningful
       when the complete reference signal has nonzero energy.

    2. ``relative_band_error`` divides by the reference energy inside that
       same band. It is returned as ``None`` when that band contains
       negligible reference energy, because the ratio would be unstable or
       scientifically misleading.
    """
    if minimum_reference_energy_fraction <= 0.0:
        raise ValueError(
            "minimum_reference_energy_fraction must be positive."
        )

    if not np.array_equal(
        prediction_spectrum.frequencies,
        reference_spectrum.frequencies,
    ):
        raise ValueError(
            "Prediction and reference spectra must use identical frequencies."
        )

    frequencies = reference_spectrum.frequencies
    prediction_coefficients = prediction_spectrum.coefficients
    reference_coefficients = reference_spectrum.coefficients

    coefficient_error = (
        prediction_coefficients - reference_coefficients
    )

    prediction_energy_per_bin = (
        np.abs(prediction_coefficients) ** 2
    )
    reference_energy_per_bin = (
        np.abs(reference_coefficients) ** 2
    )
    error_energy_per_bin = np.abs(coefficient_error) ** 2

    total_prediction_energy = float(
        np.sum(prediction_energy_per_bin)
    )
    total_reference_energy = float(
        np.sum(reference_energy_per_bin)
    )

    if total_reference_energy <= 0.0:
        raise ValueError(
            "The complete reference signal has zero spectral energy."
        )

    masks = construct_band_masks(
        frequencies=frequencies,
        bands=bands,
    )

    output: dict[str, BandMetrics] = {}

    for band_name, mask in masks.items():
        number_of_bins = int(np.count_nonzero(mask))

        if number_of_bins == 0:
            raise ValueError(
                f"The {band_name} band contains no Fourier bins. "
                "Use more spatial points or wider band limits."
            )

        reference_energy = float(
            np.sum(reference_energy_per_bin[mask])
        )
        prediction_energy = float(
            np.sum(prediction_energy_per_bin[mask])
        )
        error_energy = float(
            np.sum(error_energy_per_bin[mask])
        )

        reference_energy_fraction = (
            reference_energy / total_reference_energy
        )
        prediction_energy_fraction = (
            prediction_energy / total_prediction_energy
            if total_prediction_energy > 0.0
            else 0.0
        )

        normalized_total_error = float(
            np.sqrt(
                error_energy / total_reference_energy
            )
        )

        if (
            reference_energy_fraction
            >= minimum_reference_energy_fraction
        ):
            relative_band_error: float | None = float(
                np.sqrt(
                    error_energy / reference_energy
                )
            )
        else:
            relative_band_error = None

        band_frequencies = frequencies[mask]

        output[band_name] = BandMetrics(
            frequency_min=float(band_frequencies[0]),
            frequency_max=float(band_frequencies[-1]),
            number_of_bins=number_of_bins,
            reference_energy=reference_energy,
            prediction_energy=prediction_energy,
            error_energy=error_energy,
            reference_energy_fraction=reference_energy_fraction,
            prediction_energy_fraction=prediction_energy_fraction,
            error_normalized_by_total_reference_energy=(
                normalized_total_error
            ),
            relative_band_error=relative_band_error,
        )

    return output


def analyze_spatial_profile(
    x: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    bands: FrequencyBands,
    use_odd_extension: bool = True,
) -> dict[str, Any]:
    """Analyze one predicted and exact spatial profile."""
    prediction_spectrum = compute_spatial_fourier_spectrum(
        x=x,
        values=prediction,
        use_odd_extension=use_odd_extension,
    )
    reference_spectrum = compute_spatial_fourier_spectrum(
        x=x,
        values=reference,
        use_odd_extension=use_odd_extension,
    )

    band_metrics = calculate_band_metrics(
        prediction_spectrum=prediction_spectrum,
        reference_spectrum=reference_spectrum,
        bands=bands,
    )

    spatial_error = prediction - reference

    dominant_reference_index = int(
        np.argmax(reference_spectrum.amplitudes[1:]) + 1
    )
    dominant_prediction_index = int(
        np.argmax(prediction_spectrum.amplitudes[1:]) + 1
    )

    return {
        "prediction_spectrum": prediction_spectrum,
        "reference_spectrum": reference_spectrum,
        "band_metrics": band_metrics,
        "mean_spatial_error": float(np.mean(spatial_error)),
        "spatial_relative_l2": float(
            np.linalg.norm(spatial_error)
            / np.linalg.norm(reference)
        ),
        "dominant_reference_frequency": float(
            reference_spectrum.frequencies[
                dominant_reference_index
            ]
        ),
        "dominant_prediction_frequency": float(
            prediction_spectrum.frequencies[
                dominant_prediction_index
            ]
        ),
    }


def plot_spectrum_comparison(
    analysis: dict[str, Any],
    time_value: float,
    output_path: Path,
) -> None:
    """Save a log-scale spectrum comparison for one time."""
    prediction_spectrum: SpectrumResult = analysis[
        "prediction_spectrum"
    ]
    reference_spectrum: SpectrumResult = analysis[
        "reference_spectrum"
    ]

    positive_floor = 1.0e-16

    figure = plt.figure(figsize=(8, 5))
    axis = figure.add_subplot(111)

    axis.semilogy(
        reference_spectrum.frequencies,
        np.maximum(
            reference_spectrum.amplitudes,
            positive_floor,
        ),
        label="Exact spectrum",
    )
    axis.semilogy(
        prediction_spectrum.frequencies,
        np.maximum(
            prediction_spectrum.amplitudes,
            positive_floor,
        ),
        linestyle="--",
        label="PINN spectrum",
    )

    axis.set_xlabel("Spatial frequency")
    axis.set_ylabel("Fourier amplitude")
    axis.set_title(
        f"Spatial Fourier spectrum at t={time_value:.3f}"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def serializable_analysis(
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """Remove NumPy arrays before writing analysis results to JSON."""
    return {
        "mean_spatial_error": analysis["mean_spatial_error"],
        "spatial_relative_l2": analysis["spatial_relative_l2"],
        "dominant_reference_frequency": (
            analysis["dominant_reference_frequency"]
        ),
        "dominant_prediction_frequency": (
            analysis["dominant_prediction_frequency"]
        ),
        "nyquist_frequency": (
            analysis["reference_spectrum"].nyquist_frequency
        ),
        "spatial_step": (
            analysis["reference_spectrum"].spatial_step
        ),
        "used_odd_extension": (
            analysis["reference_spectrum"].used_odd_extension
        ),
        "bands": {
            name: asdict(metrics)
            for name, metrics
            in analysis["band_metrics"].items()
        },
    }


def analyze_saved_heat_results(
    evaluation_path: Path,
    output_directory: Path,
    requested_times: Sequence[float],
    bands: FrequencyBands,
    use_odd_extension: bool = True,
) -> dict[str, Any]:
    """Analyze saved Heat PINN predictions at selected times."""
    if not evaluation_path.exists():
        raise FileNotFoundError(
            f"Evaluation data not found: {evaluation_path}"
        )

    data = np.load(evaluation_path)

    x = np.asarray(data["x"], dtype=float)
    t = np.asarray(data["t"], dtype=float)
    prediction = np.asarray(data["prediction"], dtype=float)
    reference = np.asarray(data["reference"], dtype=float)

    expected_shape = (x.size, t.size)

    if prediction.shape != expected_shape:
        raise ValueError(
            "prediction has shape "
            f"{prediction.shape}, expected {expected_shape}."
        )
    if reference.shape != expected_shape:
        raise ValueError(
            "reference has shape "
            f"{reference.shape}, expected {expected_shape}."
        )

    output_directory.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "method": (
            "one-sided orthonormal FFT of an odd spatial extension"
            if use_odd_extension
            else "one-sided orthonormal FFT on the original interval"
        ),
        "band_definition": asdict(bands),
        "important_note": (
            "relative_band_error is null when the exact solution contains "
            "negligible energy in that band. Use "
            "error_normalized_by_total_reference_energy for stable "
            "cross-band comparisons."
        ),
        "times": {},
    }

    for requested_time in requested_times:
        time_index = int(
            np.argmin(np.abs(t - requested_time))
        )
        actual_time = float(t[time_index])

        analysis = analyze_spatial_profile(
            x=x,
            prediction=prediction[:, time_index],
            reference=reference[:, time_index],
            bands=bands,
            use_odd_extension=use_odd_extension,
        )

        time_key = f"{actual_time:.6f}"

        results["times"][time_key] = (
            serializable_analysis(analysis)
        )

        filename_time = (
            f"{actual_time:.3f}".replace(".", "p")
        )

        plot_spectrum_comparison(
            analysis=analysis,
            time_value=actual_time,
            output_path=(
                output_directory
                / f"fourier_spectrum_t_{filename_time}.png"
            ),
        )

    with (
        output_directory / "spectral_analysis.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    return results


def run_scientific_self_check() -> None:
    """Verify that the odd-extension FFT identifies the expected sine mode."""
    x = np.linspace(0.0, 1.0, 201)
    reference = np.sin(np.pi * x)

    spectrum = compute_spatial_fourier_spectrum(
        x=x,
        values=reference,
        use_odd_extension=True,
    )

    dominant_index = int(
        np.argmax(spectrum.amplitudes[1:]) + 1
    )
    dominant_frequency = float(
        spectrum.frequencies[dominant_index]
    )

    # sin(pi*x) on [0,1] becomes the fundamental mode of the odd
    # extension over a period of length 2, so f = 1/2.
    assert np.isclose(
        dominant_frequency,
        0.5,
        atol=1.0e-12,
        rtol=0.0,
    )

    identical_analysis = analyze_spatial_profile(
        x=x,
        prediction=reference.copy(),
        reference=reference,
        bands=FrequencyBands(),
        use_odd_extension=True,
    )

    for metrics in identical_analysis["band_metrics"].values():
        assert np.isclose(
            metrics.error_energy,
            0.0,
            atol=1.0e-14,
        )
        assert np.isclose(
            metrics.error_normalized_by_total_reference_energy,
            0.0,
            atol=1.0e-14,
        )

    print("Fourier-analysis scientific self-check passed.")
    print(
        "Detected dominant frequency for sin(pi*x): "
        f"{dominant_frequency:.6f}"
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate spatial Fourier spectra and low-, middle-, "
            "and high-frequency errors for Heat PINN results."
        )
    )

    parser.add_argument(
        "--evaluation-path",
        type=Path,
        default=Path(
            "outputs/heat_figures/evaluation_grid.npz"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(
            "outputs/heat_spectral_analysis"
        ),
    )
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
    )
    parser.add_argument(
        "--low-upper-fraction",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--middle-upper-fraction",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--direct-fft",
        action="store_true",
        help=(
            "Use a direct FFT on [0,1] instead of the default "
            "zero-Dirichlet odd extension."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_scientific_self_check()

    arguments = parse_arguments()

    selected_bands = FrequencyBands(
        low_upper_fraction=(
            arguments.low_upper_fraction
        ),
        middle_upper_fraction=(
            arguments.middle_upper_fraction
        ),
    )

    results = analyze_saved_heat_results(
        evaluation_path=arguments.evaluation_path,
        output_directory=arguments.output_directory,
        requested_times=arguments.times,
        bands=selected_bands,
        use_odd_extension=not arguments.direct_fft,
    )

    print("Saved Heat PINN spectral analysis.")
    print(f"Output directory: {arguments.output_directory}")

    for time_value, time_results in results["times"].items():
        print(f"\nt = {time_value}")
        print(
            "  dominant exact frequency: "
            f"{time_results['dominant_reference_frequency']:.6f}"
        )
        print(
            "  dominant PINN frequency:  "
            f"{time_results['dominant_prediction_frequency']:.6f}"
        )

        for band_name, metrics in time_results["bands"].items():
            relative_band_error = metrics[
                "relative_band_error"
            ]
            relative_text = (
                f"{relative_band_error:.6e}"
                if relative_band_error is not None
                else "undefined: negligible exact-band energy"
            )

            print(
                f"  {band_name:6s} | "
                "error/total-reference-energy="
                f"{metrics['error_normalized_by_total_reference_energy']:.6e} | "
                f"relative-band-error={relative_text}"
            )
