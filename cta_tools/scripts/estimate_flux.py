from cta_tools.plotting import preliminary
import matplotlib.pyplot as plt
import click
from pathlib import Path
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.modeling.models import create_crab_spectral_model
from gammapy.estimators import LightCurveEstimator
import astropy.units as u
from cta_tools.logging import setup_logging


log = setup_logging()


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("models_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(dir_okay=False))
@click.option("-r", "--reference", type=str, help="Crab Reference Spectrum. Crab Only!")
def main(config_path, models_path, output, reference):
    config = AnalysisConfig.read(config_path)
    analysis = Analysis(config)
    log.info(config)

    analysis.get_observations()
    log.info(analysis)
    log.info(dir(analysis))
    log.info(analysis.datasets)
    log.info(analysis.datasets[0].counts)
    analysis.get_datasets()
    analysis.read_models(models_path)

    # stacked fit and flux estimation
    analysis.run_fit()
    analysis.get_flux_points()

    # Plot flux points
    ax_sed, ax_residuals = analysis.flux_points.plot_fit()
    if reference:
        plot_kwargs = {
            "energy_range": [
                analysis.config.flux_points.energy.min,
                analysis.config.flux_points.energy.max,
            ],
            "energy_power": 2,
            "flux_unit": "erg-1 cm-2 s-1",
        }
        create_crab_spectral_model(reference).plot(
            **plot_kwargs, ax=ax_sed, label="Crab reference"
        )
        ax_sed.legend()
        ax_sed.set_ylim(1e-12, 1e-9)
    

    base_out = Path(output)
    ax_sed.get_figure().savefig(base_out.with_suffix(".pdf").as_posix())
    plt.clf()
    analysis.models.write(base_out.with_suffix(".yaml").as_posix(), overwrite=True)
    analysis.flux_points.write(
        base_out.with_suffix(".fits").as_posix(), overwrite=True
    )
    ax_excess = analysis.datasets["stacked"].plot_excess()
    ax_excess.get_figure().savefig(base_out.with_suffix(".excess.pdf").as_posix())
    plt.clf()
        
    config.datasets.stack = False
    analysis.get_observations()
    analysis.get_datasets()
    analysis.read_models(models_path)
    lc_maker_low = LightCurveEstimator(
        energy_edges=[.2, 5] * u.TeV, source=config.flux_points.source, reoptimize=False
    )
    lc_low = lc_maker_low.run(analysis.datasets)
    ax_lc = lc_low.plot(marker="o", label="1D")
    ax_lc.get_figure().savefig(base_out.with_suffix(".lc.pdf").as_posix())
    plt.clf()

if __name__ == "__main__":
    main()
