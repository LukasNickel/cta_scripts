from setuptools import setup, find_packages

setup(
    name="cta_tools",
    version=0.4,
    author="Lukas Nickel",
    packages=find_packages(),
    install_requires=[
        "astropy",
        "lstchain",
        "aict_tools",
        "numpy",
        "pandas",
        "rich",
        "ctapipe",
        "gammapy",
        "click",
        "pyirf",
        "pandas",
    ],
    license="MIT",
    entry_points={
        "console_scripts": [
            "plot_charges = cta_tools.scripts.plot_charges:main",
            "plot_dl1_features = cta_tools.scripts.plot_features:main",
            "plot_dl1_features_time = cta_tools.scripts.plot_time_evolution:main",
            "plot_dl2_features = cta_tools.scripts.plot_compare_dl2:main",
            "plot_theta2 = cta_tools.scripts.plot_theta2:main",
            "plot_dl3_events = cta_tools.scripts.plot_dl3:main",
            "plot_irfs = cta_tools.scripts.plot_irfs:main",
            "estimate_flux = cta_tools.scripts.estimate_flux:main",
        ]
    },
)
