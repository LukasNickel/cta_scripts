import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord, SkyOffsetFrame
import astropy.units as u
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator

from fact.analysis.statistics import li_ma_significance


def theta2(theta2_on, theta2_off, scaling, cut, threshold="", source="", ontime=None, ax=None, window=[0,1], bins=100):

    ax = ax or plt.gca()

    ax.hist(theta2_on, bins=bins, range=window, histtype='step', color='r', label='ON')
    ax.hist(theta2_off, bins=bins, range=window, histtype='stepfilled', color='tab:blue', alpha=0.5, label='OFF', weights=np.full_like(theta2_off, scaling))

    n_off = np.count_nonzero(theta2_off < cut)
    n_on = np.count_nonzero(theta2_on < cut)
    li_ma = li_ma_significance(n_on, n_off, scaling)
    n_exc_mean = n_on - scaling * n_off
    n_exc_std = np.sqrt(n_on + scaling**2 * n_off)

    txt = rf'''
    $N_\mathrm{{on}} = {n_on},\, N_\mathrm{{off}} = {n_off},\, \alpha = {scaling:.2f}$
    $N_\mathrm{{exc}} = {n_exc_mean:.0f} \pm {n_exc_std:.0f}$
    $S_\mathrm{{Li&Ma}} = {li_ma:.2f}$ in $T = {ontime.to_value(u.hour):.2f} \mathrm{{h}}$
    '''
    ax.text(0.5, 0.95, txt, transform=ax.transAxes, va='top', ha='center')
    if isinstance(cut, float):
        ax.axvline(cut, color='k', alpha=0.6, lw=1, ls='--')

    ax.set_xlabel(r'$\theta^2 \,\, / \,\, \mathrm{deg}^2$')
    ax.set_xlim(window)
    ax.legend()
    ax.figure.tight_layout()
    return ax
