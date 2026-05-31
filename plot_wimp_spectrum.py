"""
plot_wimp_spectrum.py
---------------------
Load pre-computed elastic WIMP-nucleus recoil spectra, scale to the LZ
exclusion cross-section, and plot:
  left col  : dN/dEr spectra (top) + A_d (bottom)   <- main
  right col : LZ exclusion curve (top) + legend (bottom)

Output
------
figures/WIMP_upperlimit.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
import astropy.units as u
import astropy.constants as const

from plot_setup import get_WIMPmass_colors
from get_upperlimit_spectrum import get_WIMP_pdf, load_exclusion_curve, get_exclusion_xsec
from DM_modulation_model import get_SHM_Ads


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET       = "Xe"
M_WIMPS      = [6, 10, 20, 50, 100, 500, 1000, 10_000] * u.GeV
A            = 131.293


XSEC_FILE    = "WIMP_xsec_LZ_excludedtupperlimit.txt"
XSEC_DEFAULT = 4.4e-45 * u.cm**2   # fallback for masses outside tabulated range

AD_ECC       = 0.03342

OUTPUT_DIR   = "figures"
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "WIMP_upperlimit.png")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(m_N, masses, xsecs, interp_fn):
    fig = plt.figure(figsize=(13, 6))

    gs_outer  = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.38)
    gs_left   = gs_outer[0].subgridspec(2, 1, height_ratios=[3, 2], hspace=0.08)
    gs_right  = gs_outer[1].subgridspec(2, 1, height_ratios=[3, 2], hspace=0.55)

    ax_spec   = fig.add_subplot(gs_left[0])
    ax_Ad     = fig.add_subplot(gs_left[1], sharex=ax_spec)
    ax_xsec   = fig.add_subplot(gs_right[0])
    ax_legend = fig.add_subplot(gs_right[1])

    patches = []
    labels  = []

    xsec_at_mass = {m: get_exclusion_xsec(m, interp_fn, fallback=XSEC_DEFAULT) for m in M_WIMPS}

    for m_wimp in M_WIMPS:
        sigma_f  = xsec_at_mass[m_wimp]
        mass_str = f"{int(m_wimp.value)}{m_wimp.unit}"
        color    = get_WIMPmass_colors(mass_str)

        # dot on exclusion curve
        ax_xsec.scatter(m_wimp.value, sigma_f.value, color=color, zorder=5, s=40)

        Er_wimp, pdf_wimp = get_WIMP_pdf(m_wimp, sigma_f)
        if Er_wimp is None:
            continue

        ax_spec.loglog(Er_wimp, pdf_wimp, color=color, lw=2)

        Ad_arr = get_SHM_Ads(Er_wimp, m_wimp, m_N,
                              vc=220 * u.km / u.s,
                              vesc=553 * u.km / u.s,
                              xp=0.89)
        mask = Ad_arr != 0
        ax_Ad.semilogx(Er_wimp[mask], Ad_arr[mask], color=color, lw=2)

        patches.append(Patch(color=color))
        labels.append(
            rf"M$_{{\chi}}$={int(m_wimp.value)}{m_wimp.unit}"
            rf", $\sigma$={sigma_f.value:.3g} cm$^2$"
        )

    # --- exclusion curve: solid black from file ---
    ax_xsec.loglog(masses, xsecs, color="black", lw=1.5)

    # dashed segment: 6 GeV (fallback) to 10 GeV (interpolated)
    m_6  = M_WIMPS[0]; s_6  = xsec_at_mass[m_6]
    m_10 = M_WIMPS[1]; s_10 = xsec_at_mass[m_10]
    ax_xsec.loglog([m_6.value, m_10.value], [s_6.value, s_10.value],
                   color="black", lw=1.5, ls="--")

    ax_xsec.set_xlabel(r"M$_{\chi}$ [GeV]", fontsize=11)
    ax_xsec.set_ylabel(r"$\sigma_n$ [cm$^2$]", fontsize=11)
    ax_xsec.set_title("LZ WS2024", fontsize=10)
    ax_xsec.tick_params(labelsize=9)
    ax_xsec.grid(True, which="both", ls="--", alpha=0.4)
    ax_xsec.spines["top"].set_visible(False)
    ax_xsec.spines["right"].set_visible(False)

    # --- spectrum panel ---
    ax_spec.set_ylabel(r"dN/dE$_r$ [ton$^{-1}$ yr$^{-1}$ keV$^{-1}$]", fontsize=13)
    ax_spec.set_ylim(1e-5, 1e4)
    ax_spec.grid(True, ls="--", alpha=0.4)
    ax_spec.spines["top"].set_visible(False)
    ax_spec.spines["right"].set_visible(False)
    plt.setp(ax_spec.get_xticklabels(), visible=False)

    # --- A_d panel ---
    ax_Ad.set_xlabel(r"nuclear recoil energy $E_r$ [keV]", fontsize=13)
    ax_Ad.set_ylabel(r"A$_{d, SHM}$", fontsize=13)
    ax_Ad.set_xlim(0.01, 90)
    ax_Ad.set_ylim(-0.05, 0.1)
    ax_Ad.axhline(y=AD_ECC, lw=2, ls=":", color="black")
    ax_Ad.text(0.01, AD_ECC, r"A$_{d,\nu}$", ha="left", va="bottom", fontsize=11)
    ax_Ad.axhspan(-0.05, 0, alpha=0.2, color="grey")
    ax_Ad.text(0.01, 0., "phase reverse", color="grey",
               ha="left", va="top", fontsize=11)
    ax_Ad.grid(True, which="both", ls="--", alpha=0.4)
    ax_Ad.spines["top"].set_visible(False)
    ax_Ad.spines["right"].set_visible(False)
    ax_Ad.tick_params(which="major", length=1)
    ax_Ad.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.5g"))

    # --- legend panel ---
    ax_legend.axis("off")
    ax_legend.legend(patches, labels,
                     loc="upper center", ncol=1, fontsize=9, frameon=False)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    m_N = A * (const.m_n * const.c**2).to(u.GeV)

    masses, xsecs, interp_fn = load_exclusion_curve(XSEC_FILE)

    fig = make_plot(m_N, masses, xsecs, interp_fn)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    print(f"Saved -> {OUTPUT_FILE}")
    plt.close(fig)
