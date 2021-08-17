import matplotlib.axes
from matplotlib.offsetbox import AnchoredText

class MyAxes(matplotlib.axes.Axes):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        ab = AnchoredText("Preliminary", loc="lower right", frameon=False,
                          borderpad=0, prop=dict(alpha=0.3, fontsize=20))
        ab.set_zorder(0)
        self.add_artist(ab)

matplotlib.axes.Axes = MyAxes