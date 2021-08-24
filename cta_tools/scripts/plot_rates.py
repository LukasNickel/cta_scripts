from cta_tools.plotting import preliminary
from cta_tools.plotting.features import plot_binned_time_evolution
import pandas as pd
import click
import numpy as np
from pyirf.binning import create_bins_per_decade
import matplotlib.pyplot as plt
from pathlib import Path
import astropy.units as u
from tqdm import tqdm
from astropy.table import vstack
from cta_tools.io import read_lst_dl1, read_mc_dl1, read_sim_info, save_plot_data, read_plot_data
from cta_tools.plotting.features import compare_rates
import matplotlib
from pyirf.spectral import (
    calculate_event_weights,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
    PowerLaw,
)

if matplotlib.get_backend() == "pgf":
    from matplotlib.backends.backend_pgf import PdfPages
else:
    from matplotlib.backends.backend_pdf import PdfPages


# keys that are used to store data
keys = {
    "intensity_hist": "intensity",
    "energy_hist": "energy",
    "event_rate_plot": "rate_over_time",
}

data_structure = {
    key: {"bins": None, "values": None} for key in keys.values()
}


def load_data(files, cache):
    if cache.exists():
        plot_values = read_plot_data(cache, data_structure)
    else:
        proton_file = files["protons"]
        electron_file = files["electrons"]
        observation_files = files["observations"]
        plot_values = {}
        obstime = 0 * u.s
        observations = {}
        for f in tqdm(observation_files):
            data = read_lst_dl1(f)
            observations[data[0]["obs_id"]] = data
            run_time = (data["time"][-1] - data["time"][0]).to(u.s)
            obstime += run_time
        combined = vstack(list(observations.values()))
        combined["weights"] = 1 / obstime.to_value(u.s)

        proton_sim_info = read_sim_info(proton_file)
        protons = read_mc_dl1(proton_file)

        protons["weights"] = calculate_event_weights(
            protons["true_energy"],
            IRFDOC_PROTON_SPECTRUM,
            PowerLaw.from_simulation(proton_sim_info, 1*u.s),
        )
        electron_sim_info = read_sim_info(electron_file)
        electrons = read_mc_dl1(electron_file)
        electrons["weights"] = calculate_event_weights(
            electrons["true_energy"],
            IRFDOC_ELECTRON_SPECTRUM,
            PowerLaw.from_simulation(electron_sim_info, 1*u.s),
        )
        background = vstack([protons, electrons])
        
        # intensity
        int_bins = np.logspace(np.log10(background['hillas_intensity'].min()), np.log10(background['hillas_intensity'].max()), 30)
        int_counts_obs, _ = np.histogram(combined["hillas_intensity"], weights = combined["weights"], bins=int_bins)
        int_counts_mc, _ = np.histogram(background["hillas_intensity"], weights = background["weights"], bins=int_bins)
        #store.put(f"{keys['intensity_hist']}/bins", int_bins)
        int_df = pd.DataFrame({"Observations":int_counts_obs, "MC": int_counts_mc})
        #store.put(f"{keys['intensity_hist']}/values", int_df)
        plot_values[f"{keys['intensity_hist']}"] = {
            "bins": pd.Series(int_bins),
            "values": int_df,
        }

        if "gamma_energy_prediction" in combined.columns:
            energy_bins = create_bins_per_decade(50*u.GeV, 10*u.TeV, 5)
            energy_counts_obs, _ = np.histogram(combined["gamma_energy_prediction"], weights = combined["weights"], bins=energy_bins)
            energy_counts_mc, _ = np.histogram(background["gamma_energy_prediction"], weights = background["weights"], bins=energy_bins)

            #store.put(f"{keys['energy_hist']}/bins", energy_bins)
            energy_df = pd.DataFrame({"Observations":energy_counts_obs, "MC": energy_counts_mc})
            #store.put(f"{keys['energy_hist']}/values", energy_df)
            plot_values[f"{keys['energy_hist']}"] = {
                "bins": pd.Series(energy_bins),
                "values": energy_df,
            }
        else:
            plot_values[f"{keys['energy_hist']}"] = {}

        combined["delta_t_sec"] = (combined["time"] - combined["time"][0]).sec
        # more time later of course
        last = combined["delta_t_sec"].max()
        time_bins = np.linspace(0, last, 20)
        rate_over_time, _ = np.histogram(combined["delta_t_sec"], bins=time_bins, weights=combined["weights"])
        #store.put(f"{keys['event_rate_plot']}/bins", time_bins)
        #rate_df = pd.DataFrame({"Event rate": rate_over_time})
        #store.put(f"{keys['event_rate_plot']}/values", rate_df)
        binned_rate = pd.DataFrame(
            {
                "center": (time_bins[1:] + time_bins[:-1])/2,
                "mean": rate_over_time,
                "width": np.diff(time_bins),
                "std": np.sqrt(rate_over_time),
            }
        )

        plot_values[f"{keys['event_rate_plot']}"] = {
            "bins": pd.Series(time_bins),
            "values": binned_rate,
        }
        save_plot_data(cache, plot_values)
    return plot_values


# make this a yaml file or smth for the file lists and spectras to weight to
@click.command()
@click.argument(
    "input_files",
    nargs=-1,
)
@click.option("--protons", "-p", type=click.Path(exists=True))
@click.option("--electrons", "-e", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(exists=False, dir_okay=False))
def main(
    input_files,
    protons,
    electrons,
    output,
):
    cache = Path(output).with_suffix(".h5")


    plot_data = load_data({"observations": input_files, "protons": protons, "electrons": electrons}, cache)

    figs = []
    figs.append(compare_rates(
        plot_data[keys["intensity_hist"]]["values"]["Observations"],
        plot_data[keys["intensity_hist"]]["values"]["MC"],
        plot_data[keys["intensity_hist"]]["bins"].values,
    ))
    figs[-1].axes[0].set_title("Event rates")
    figs[-1].axes[1].set_xlabel("Hillas intensity")

    if plot_data[keys["energy_hist"]]:
        figs.append(compare_rates(
            plot_data[keys["energy_hist"]]["values"]["Observations"],
            plot_data[keys["energy_hist"]]["values"]["MC"],
            plot_data[keys["energy_hist"]]["bins"].values,
        ))
        figs[-1].axes[0].set_title("Event rates")
        figs[-1].axes[1].set_xlabel("Energy [GeV]")



    fig, ax = plt.subplots()
    plot_binned_time_evolution(plot_data[keys["event_rate_plot"]]["values"], ax=ax)
    figs.append(fig)

    if output is None:
        plt.show()
    else:
        with PdfPages(output) as pdf:
            for fig in figs:
                fig.tight_layout(pad=0)
                pdf.savefig(fig)
