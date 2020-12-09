import click
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from regions import CircleSkyRegion

from gammapy.data import DataStore
from gammapy.modeling import Fit
from gammapy.estimators import FluxPointsEstimator
from gammapy.maps import Map, MapAxis
from gammapy.datasets import (
    Datasets,
    FluxPointsDataset,
    SpectrumDataset,
)
from gammapy.modeling.models import (
    create_crab_spectral_model,
    SkyModel,
    LogParabolaSpectralModel
)
from gammapy.makers import (
    SpectrumDatasetMaker,
    ReflectedRegionsBackgroundMaker,
)


def main():
    data_store = DataStore.from_dir(
        "/run/media/lukas/media/cta/lst/dl3_local/",
        hdu_table_filename="hdu-index.fits.gz",
        obs_table_filename="obs-index.fits.gz"
    )
    selection = dict(
        type="sky_circle",
        frame="icrs",
        lon="83.633 deg",
        lat="22.014 deg",
        radius="5 deg",
    )
    selected_obs_table = data_store.obs_table.select_observations(selection)
    observations = data_store.get_observations(selected_obs_table['OBS_ID'])

    target_position = SkyCoord(ra=83.63, dec=22.01, unit="deg", frame="icrs")
    on_region_radius = Angle("0.11 deg")
    on_region = CircleSkyRegion(
        center=target_position,
        radius=on_region_radius
    )
    exclusion_region = CircleSkyRegion(
        center=SkyCoord(183.604, -8.708, unit="deg", frame="galactic"),
        radius=0.5 * u.deg,
    )

    skydir = target_position.galactic
    exclusion_mask = Map.create(
        npix=(150, 150), binsz=0.05, skydir=skydir, proj="TAN", frame="icrs"
    )

    mask = exclusion_mask.geom.region_mask([exclusion_region], inside=False)
    exclusion_mask.data = mask
    e_reco = MapAxis.from_energy_bounds(
        0.1, 30, 15, unit="TeV", name="energy"
    )
    e_true = MapAxis.from_energy_bounds(
        0.1, 30, 15, unit="TeV", name="energy_true"
    )
    dataset_empty = SpectrumDataset.create(
        e_reco=e_reco, e_true=e_true, region=on_region
    )

    dataset_maker = SpectrumDatasetMaker(
        containment_correction=False, selection=["counts", "exposure", "edisp", "aeff"]
    )
    bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)

    datasets = Datasets()

    for obs_id, observation in zip(observations.ids, observations):
        dataset = dataset_maker.run(
            dataset_empty.copy(name=str(obs_id)), observation
        )
        dataset_on_off = bkg_maker.run(dataset, observation)
        datasets.append(dataset_on_off)

    spectral_model = LogParabolaSpectralModel(
        alpha=2, beta=8, amplitude=3e-11 * u.Unit("cm-2 s-1 TeV-1"), reference=1*u.TeV
    )
    model = SkyModel(spectral_model=spectral_model, name="crab")

    for dataset in datasets:
        dataset.models = model

    fit_joint = Fit(datasets)
    fit_joint.run(backend="scipy")
    model_best_joint = model.copy()

    e_min, e_max = 0.1, 30
    energy_edges = np.logspace(np.log10(e_min), np.log10(e_max), 16) * u.TeV
    fpe = FluxPointsEstimator(energy_edges=energy_edges, source="crab")
    flux_points = fpe.run(datasets=datasets)
    flux_points_dataset = FluxPointsDataset(
        data=flux_points, models=model_best_joint
    )

    plt.figure()
    ax = plt.gca()
    flux_points_dataset.plot_spectrum(ax)

    if plot_reference:
        plot_kwargs = {
            "energy_range": [e_min, e_max] * u.TeV,
            "energy_power": 2,
            "flux_unit": "erg-1 cm-2 s-1",
        }
        create_crab_spectral_model("magic_lp").plot(
            **plot_kwargs, ax=ax, label="Crab reference [MAGIC]"
        )
    plt.legend()
    plt.savefig('build/fluxpoints_all.pdf')


if __name__ == '__main__':
    main()
