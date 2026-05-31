"""
check_gvmin.py
--------------
Compute and plot the mean inverse speed g(v_min) for a set of DM masses,
then save the figure for visual validation against Fig. 19 of arXiv:1211.7090.

Output
------
check_gvmin.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const

from scatter_kinematics import momentum
from scatter_target import get_target_info
from DM_halo import get_vmin, get_gvmin


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET    = "Xe"
M_CHIS    = [6 * u.GeV, 10 * u.GeV, 50 * u.GeV]

V_ESC     = 533 * u.km / u.s
SIGMA_V   = 270 * u.km / u.s
V_EARTH   = 232 * u.km / u.s
V_UNIT    = u.km / u.s
ER_UNIT   = u.keV

ER_START  = 0.01 * u.keV
ER_NBINS  = 101

OUTPUT_DIR   = "figures"
OUTPUT    = os.path.join(OUTPUT_DIR, "check_gvmin.png")



# ---------------------------------------------------------------------------
# Kinematics helpers
# ---------------------------------------------------------------------------

def _ermax(e_chi, m_chi, m_N):
    """Kinematic maximum recoil energy (elastic, non-rel fallback)."""
    if e_chi <= m_chi:
        return 0 * ER_UNIT
    v_chi = np.sqrt(e_chi**2 - m_chi**2) / e_chi
    if v_chi <= 0.01:                          # non-relativistic
        mu = m_chi * m_N / (m_chi + m_N)
        return (2 * mu**2 * v_chi**2 / m_N).to(ER_UNIT)
    p     = momentum(e_chi, m_chi)
    p_out = momentum(e_chi, m_chi)             # elastic: dE = 0
    return ((p + p_out)**2 / (2 * m_N)).to(ER_UNIT)


def er_grid(m_chi, m_N):
    """Log-spaced Er grid up to the kinematic maximum."""
    v_max      = V_ESC + V_EARTH
    gamma_max  = 1 / np.sqrt(1 - (v_max / const.c.to(V_UNIT))**2)
    er_max     = _ermax(gamma_max * m_chi, m_chi, m_N)
    er_end     = np.ceil(er_max.value) * ER_UNIT
    return np.logspace(
        np.log10(ER_START.value), np.log10(er_end.value), ER_NBINS
    ) * ER_UNIT


# ---------------------------------------------------------------------------
# Compute g(v_min) curves
# ---------------------------------------------------------------------------

def compute_gvmin_curves(m_N):
    """Return {m_chi: (Ers, g_values)} and g(0)."""
    g0, g_unit = get_gvmin(0 * V_UNIT, V_EARTH,
                            v_esc=V_ESC, sigma_v=SIGMA_V, v_unit=V_UNIT)

    curves = {}
    for m_chi in M_CHIS:
        Ers   = er_grid(m_chi, m_N)
        gs    = []
        for Er in Ers:
            vmin       = get_vmin(m_N, m_chi, Er, 0 * m_chi.unit)
            g_val, _   = get_gvmin(vmin, V_EARTH,
                                    v_esc=V_ESC, sigma_v=SIGMA_V, v_unit=V_UNIT)
            gs.append(g_val.value)
        curves[m_chi] = (Ers, np.array(gs) * g_unit)

    return curves, g0, g_unit


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_figure(curves, g0, g_unit, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Mean inverse speed g(v_min) — {TARGET}\n"
        "Ref: Fig. 19 arXiv:1211.7090",
        fontsize=12
    )

    for m_chi, (Ers, gs) in curves.items():
        label = str(m_chi)
        axes[0].loglog(Ers.value, gs.value,    marker="o", ms=3, label=label)
        axes[1].loglog(Ers.value, gs / g0,     marker="o", ms=3, label=label)

    # --- left panel: g(v_min) ---
    axes[0].set_xlabel("nuclear recoil energy [keV]", fontsize=13)
    axes[0].set_ylabel(f"g(v_min)  [{g_unit}]",       fontsize=13)
    axes[0].set_ylim(1e-7, 1e-1)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, which="both", ls="--", alpha=0.4)

    # --- right panel: g(v_min) / g(0) ---
    axes[1].set_xlabel("nuclear recoil energy [keV]", fontsize=13)
    axes[1].set_ylabel("g(v_min) / g(0)",              fontsize=13)
    axes[1].set_xlim(ER_START.value, 100)
    axes[1].set_ylim(1e-3, 1.1)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, which="both", ls="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved → {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    A, Z = get_target_info(TARGET)
    m_N  = A * (const.m_n * const.c**2).to(u.GeV)

    curves, g0, g_unit = compute_gvmin_curves(m_N)
    make_figure(curves, g0, g_unit, OUTPUT)
