import matplotlib.pyplot as plt
import click
from pathlib import Path
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.modeling.models import create_crab_spectral_model


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("models_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", type=click.Path(exists=True, file_okay=False))
@click.option("-r", "--reference", type=str, help="Crab Reference Spectrum. Crab Only!")
def main(config_path, models_path, output, reference):
    config = AnalysisConfig.read(config_path)
    analysis = Analysis(config)
    analysis.get_observations()
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
    if output:
        out = Path(output)
        plt.savefig(out / f"{config.flux_points.source}_flux_points.pdf")
        analysis.models.write(
            out / f"{config.flux_points.source}_model.yaml", overwrite=True
        )
        analysis.flux_points.write(
            out / f"{config.flux_points.source}_flux.fits", overwrite=True
        )

    else:
        plt.show()


if __name__ == "__main__":
    main()
