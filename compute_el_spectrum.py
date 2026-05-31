"""
compute_el_spectrum.py
----------------------
Compute the elastic WIMP-nucleus nuclear recoil spectrum dN/dEr and save
to a tab-separated text file.

Output file
-----------
WIMP_N_el_spectra/WIMP-Nel_<mass><unit>-<target>_<xsec>_pdf.txt

Usage
-----
Edit the parameters under "Configuration" and run:
    python compute_el_spectrum.py
"""

import os
import numpy as np
import astropy.units as u
import astropy.constants as const

from scatter_kinematics import ermax
from scatter_target import get_target_info, get_target_dNT_dM
from DM_halo import get_vmin, get_gvmin
from scatter_xsec import xsecEr_WIMPN_el


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET       = "Xe"
M_CHIS       = [6, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500, 1000, 5000, 10000] * u.GeV   # DM masses to scan
XSEC_WIMPN   = 1e-45  * u.cm**2    # WIMP-nucleon cross-section

RHO          = 0.3   * u.GeV / u.cm**3   # local DM density
V_ESC        = 533   * u.km / u.s        # galactic escape speed
SIGMA_V      = 270   * u.km / u.s        # velocity dispersion
V_EARTH      = 232   * u.km / u.s        # Earth speed (galactic frame)

ER_START     = 0.01  * u.keV
ER_NBINS     = 501

OUTPUT_DIR   = "WIMP_N_el_spectra"


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

V_UNIT       = u.km / u.s
ER_UNIT      = u.keV
RATE_UNIT    = 1 / u.tonne / u.yr / u.keV


# ---------------------------------------------------------------------------
# Target & DM-nucleus kinematics
# ---------------------------------------------------------------------------

def build_er_grid(m_chi, m_N, er_start, n_bins):
    """Log-spaced Er grid from er_start up to the kinematic maximum."""
    v_max       = V_ESC + V_EARTH
    gamma_vmax  = 1 / np.sqrt(1 - (v_max / const.c.to(V_UNIT))**2)
    e_chi_max   = gamma_vmax * m_chi
    er_max      = ermax(e_chi_max, m_chi, 0 * u.keV, m_N).to(ER_UNIT)
    er_end      = np.ceil(er_max.value) * ER_UNIT
    return np.logspace(
        np.log10(er_start.value), np.log10(er_end.value), n_bins
    ) * ER_UNIT


# ---------------------------------------------------------------------------
# Spectrum calculation
# ---------------------------------------------------------------------------

def compute_spectrum(m_chi, m_N, mu_wimpN, dN_T_dM, Ers):
    """
    Return dN/dEr [RATE_UNIT] at each Er in Ers.

    dN/dEr = (dN_T/dM) · m_N c² · ρ / (2 μ² m_χ) · σ(Er) · g(v_min)

    References
    ----------
    arXiv:astro-ph/0307190 Eq. 2 / arXiv:1209.3339 Eq. 3
    """
    rates = np.empty(len(Ers))

    for i, Er in enumerate(Ers):
        vmin = get_vmin(m_N, m_chi, Er, 0 * m_chi.unit)
        g_vmin, _ = get_gvmin(vmin, V_EARTH,
                               v_esc=V_ESC, sigma_v=SIGMA_V, v_unit=V_UNIT)

        xsec    = xsecEr_WIMPN_el(A, m_chi, Er,
                                   unit_convert=(const.hbar * const.c).to(u.keV * u.fm),
                                   xsec_WIMPnucleon=XSEC_WIMPN,
                                   s=0.9 * u.fm, r_0=0.52 * u.fm)
        xsec_Er = xsec.xsec_Er()

        dN_dEr = (dN_T_dM * m_N * const.c**2
                  * RHO / (2 * mu_wimpN**2 * m_chi)
                  * xsec_Er * g_vmin).to(RATE_UNIT)

        rates[i] = dN_dEr.value

    return rates * RATE_UNIT


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def output_filename(m_chi, target, xsec):
    xsec_str = str(xsec).replace(" ", "")
    mass_str = f"{int(m_chi.value)}{m_chi.unit}"
    return f"WIMP-Nel_{mass_str}-{target}_{xsec_str}_pdf.txt"


def save_spectrum(Ers, rates, filepath):
    """Write a two-column tab-separated file with a header line."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    header = f"Er [{Ers.unit}]\t\tdN_dEr [{rates.unit}]"
    data   = np.column_stack([Ers.value, rates.value])
    with open(filepath, "w") as f:
        f.write(header + "\n")
        np.savetxt(f, data, delimiter="\t", fmt="%.6e")
    print(f"\nSaved → {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    A, Z    = get_target_info(TARGET)
    m_N     = A * (const.m_n * const.c**2).to(u.GeV)
    dN_T_dM = get_target_dNT_dM(A, abundance=1.0)

    print(f"Target : {TARGET}  (A={A}, Z={Z})")
    print(f"σ_n    : {XSEC_WIMPN}")
    print(f"Masses : {M_CHIS}\n")

    for m_chi in M_CHIS:
        mu_wimpN = m_chi * m_N / (m_chi + m_N)
        Ers      = build_er_grid(m_chi, m_N, ER_START, ER_NBINS)

        print(f"  m_chi = {m_chi}  ({len(Ers)} bins, Er_max = {Ers[-1]:.2f})")
        rates = compute_spectrum(m_chi, m_N, mu_wimpN, dN_T_dM, Ers)

        fname    = output_filename(m_chi, TARGET, XSEC_WIMPN)
        filepath = os.path.join(OUTPUT_DIR, fname)
        save_spectrum(Ers, rates, filepath)
