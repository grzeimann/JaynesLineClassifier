from .simple import SkyBox, plot_distributions, plot_label_distribution_comparison, plot_probability_circle
from .model_ppp import skybox_solid_angle_sr
from .pipeline import NoiseCubeReader, LambdaSliceSpaxelIndex, wcs_from_indices, build_catalog_table
from .noise_histogram import NoiseHistogram, build_noise_histogram
from .completeness_providers import (
    CompletenessProvider,
    CatalogCompleteness,
    NoiseHistogramCompleteness,
    NoiseBinConditionedCompleteness,
)
from .rate_integrand import rate_density_integrand_per_flux
from .orchestrator import (
    KernelEnv,
    draw_signal_and_flux_stub,
    expected_counts_per_cell,
    sample_counts,
    simulate_sources_for_cell,
    run_simulation,
)
from .exp_plots import write_experimental_plots
# Kernel API (experimental)
from .kernel import (
    GaussianLineProfile,
    SkewGaussianLineProfile,
    OIIDoubletProfile,
    LineProfile,
    draw_signal_and_flux as kernel_draw_signal_and_flux,
    build_profile_from_prior,
)

__all__ = [
    "SkyBox",
    "plot_distributions",
    "plot_label_distribution_comparison",
    "plot_probability_circle",
    "skybox_solid_angle_sr",
    # Experimental simulation pipeline scaffolding
    "NoiseCubeReader",
    "LambdaSliceSpaxelIndex",
    "wcs_from_indices",
    "build_catalog_table",
    # Noise histogram
    "NoiseHistogram",
    "build_noise_histogram",
    # Completeness providers
    "CompletenessProvider",
    "CatalogCompleteness",
    "NoiseHistogramCompleteness",
    "NoiseBinConditionedCompleteness",
    # Rate integrand
    "rate_density_integrand_per_flux",
    # Orchestrator and kernel stub
    "KernelEnv",
    "draw_signal_and_flux_stub",
    "expected_counts_per_cell",
    "sample_counts",
    "simulate_sources_for_cell",
    "run_simulation",
    # Kernel API (experimental)
    "GaussianLineProfile",
    "SkewGaussianLineProfile",
    "OIIDoubletProfile",
    "LineProfile",
    "kernel_draw_signal_and_flux",
    "build_profile_from_prior",
    # Experimental plotting
    "write_experimental_plots",
]
