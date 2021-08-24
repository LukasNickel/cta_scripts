import numpy as np


def add_line_to_cam(x, y, psi, geom, ax, lw=2, lst="-", c="white", extend=(0, 0)):
    # add a line to a camera display
    min_x = geom.pix_x.min().value
    max_x = geom.pix_x.max().value
    min_y = geom.pix_y.min().value
    max_y = geom.pix_y.max().value

    xs = np.cos(np.deg2rad(psi))
    ys = np.sin(np.deg2rad(psi))

    lxmin = np.abs((x - min_x) / np.cos(np.deg2rad(psi)))
    lymin = np.abs((y - min_y) / np.sin(np.deg2rad(psi)))
    lxmax = np.abs((x - max_x) / np.cos(np.deg2rad(psi)))
    lymax = np.abs((y - max_y) / np.sin(np.deg2rad(psi)))

    # compute bounds
    if xs / ys > 0:  # positive slope
        l2 = min(lxmin, lymin)
        l1 = min(lxmax, lymax)
    else:
        l1 = min(lxmin, lymax)
        l2 = min(lxmax, lymin)
    l1 += extend[0]
    l2 += extend[1]

    x_ = [x - l1 * np.cos(np.deg2rad(psi)), x + l2 * np.cos(np.deg2rad(psi))]
    y_ = [y - l1 * np.sin(np.deg2rad(psi)), y + l2 * np.sin(np.deg2rad(psi))]

    line = ax.plot(x_, y_, linewidth=lw, linestyle=lst, color=c, clip_on=False)
