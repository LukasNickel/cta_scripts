from setuptools import setup, find_packages

setup(
    name="cta_tools",
    version=0.3,
    author="Lukas Nickel",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "lst_aict_convert = cta_tools.scripts.file_convert:main",
            "plot_theta2 = cta_tools.scripts.plot_theta2:main",
            "plot_dl3 = cta_tools.scripts.plot_dl3:main",
            "plot_irfs = cta_tools.scripts.plot_irfs:main",
            "plot_pointings = cta_tools.scripts.plot_pointings:main",
            "plot_features = cta_tools.scripts.plot_features:main",
            "mc_data_comparisons = cta_tools.scripts.mc_data_comparisons:main",
            "create_irfs = cta_tools.scripts.create_irfs:main",
            "create_event_list = cta_tools.scripts.create_event_list:main",
            "evaluate_cuts = cta_tools.scripts.evaluate_cuts:main",
            "estimate_flux = cta_tools.scripts.estimate_flux:main",
            "add_delta_t = cta_tools.scripts.add_delta_t:main",
        ]
    },
)
