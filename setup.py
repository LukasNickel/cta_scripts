from setuptools import setup, find_packages

setup(
    name='cta_tools',
    version=0.1,
    author='Lukas Nickel',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'lst_aict_convert = cta_tools.scripts.file_convert:main',
            'plot_theta2 = cta_tools.scripts.plot_theta2:main',
            'plot_irfs = cta_tools.scripts.plot_irfs:main',
            'create_irfs = cta_tools.scripts.create_irfs:main',
            'create_event_list = cta_tools.scripts.create_event_list:main',
        ]
    }
)
