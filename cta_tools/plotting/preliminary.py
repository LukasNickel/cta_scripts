import matplotlib.axes
from matplotlib.offsetbox import AnchoredText
from matplotlib.patheffects import Stroke, Normal

class PreliminaryAxes(matplotlib.axes.Axes):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        # if "main" axes 
        if not kwargs.get("sharex"):
            text = AnchoredText(
                "PRELIMINARY",
                loc="center",
                frameon=False,
                borderpad=0,
                prop=dict(
                    alpha=0.15,
                    fontsize=50,
                    rotation=30,
                    color="black"
                )
            )
            text.set_path_effects(
                [
                    Stroke(linewidth=3, foreground="white"),
                    Normal()
                ]
            )
            text.set_zorder(1000)
            self.add_artist(text)

matplotlib.axes.Axes = PreliminaryAxes
