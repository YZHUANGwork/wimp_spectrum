"""
get_detected_spectrum.py
------------------------
Load pre-computed WIMP-nucleus elastic recoil spectra and apply
cross-section scaling.  Also provides helpers to load and interpolate
experimental exclusion curves.
"""

import os
import numpy as np
import astropy.units as u
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _read(filepath):
    """Read a two-column spectrum file, skipping all non-numeric header lines."""
    with open(filepath) as f:
        lines = f.readlines()
    skiprows = 0
    for line in lines:
        try:
            float(line.split()[0])
            break
        except (ValueError, IndexError):
            skiprows += 1
    data = np.loadtxt(filepath, skiprows=skiprows)
    return data[:, 0], data[:, 1]


# ---------------------------------------------------------------------------
# Exclusion curve
# ---------------------------------------------------------------------------

def load_exclusion_curve(filepath,
                         mass_unit=u.GeV,
                         xsec_unit=u.cm**2):
    """
    Load a two-column exclusion-limit file and return an interpolator.

    Parameters
    ----------
    filepath  : path to the text file (m_chi, σ)
    mass_unit : astropy unit for the mass column  (default GeV)
    xsec_unit : astropy unit for the σ column     (default cm²)

    Returns
    -------
    masses : Quantity array
    xsecs  : Quantity array
    interp : callable — σ(m_chi), takes a plain float in mass_unit,
             raises ValueError outside the tabulated range
    """
    masses_val, xsecs_val = _read(filepath)
    masses = masses_val * mass_unit
    xsecs  = xsecs_val  * xsec_unit
    interp = interp1d(masses_val, xsecs_val)
    return masses, xsecs, interp


def get_exclusion_xsec(m_wimp, interp,
                       fallback=4.4e-45 * u.cm**2,
                       xsec_unit=u.cm**2):
    """
    Return the excluded cross-section at m_wimp from an interpolator.

    Falls back to `fallback` when m_wimp is outside the tabulated range.

    Parameters
    ----------
    m_wimp    : DM mass (astropy Quantity)
    interp    : interpolator returned by load_exclusion_curve
    fallback  : cross-section to use when outside range (default 4.4e-45 cm²)
    xsec_unit : unit of the interpolator's output (default cm²)

    Returns
    -------
    sigma : astropy Quantity [cm²]
    """
    try:
        return float(interp(m_wimp.to(u.GeV).value)) * xsec_unit
    except ValueError:
        return fallback


# ---------------------------------------------------------------------------
# Spectrum loader
# ---------------------------------------------------------------------------

def get_WIMP_pdf(m_wimp, sigma_f,
                 target="Xe",
                 sigma_i=1e-45 * u.cm**2,
                 folder="WIMP_N_el_spectra",
                 Er_unit=u.keV,
                 dN_dEr_unit=1 / u.tonne / u.yr / u.keV):
    """
    Load a pre-computed elastic spectrum and rescale to cross-section sigma_f.

    The stored spectrum was computed at sigma_i; the rescaling is linear.

    Parameters
    ----------
    m_wimp     : DM mass (astropy Quantity)
    sigma_f    : target cross-section (astropy Quantity)
    target     : element string, e.g. 'Xe'
    sigma_i    : cross-section the file was computed at (default 1e-45 cm²)
    folder     : directory containing the spectrum files
    Er_unit    : energy unit for the returned Er array
    dN_dEr_unit: rate unit for the returned spectrum

    Returns
    -------
    Er     : Quantity array [Er_unit]
    dN_dEr : Quantity array [dN_dEr_unit], scaled to sigma_f
             Returns None, None if the file is not found.
    """
    mass_str  = f"{int(m_wimp.to(u.GeV).value)}GeV"
    xsec_str  = f"{sigma_i.value}{sigma_i.unit}".replace(" ", "")
    file_name = f"WIMP-Nel_{mass_str}-{target}_{xsec_str}_pdf.txt"
    filepath  = os.path.join(folder, file_name)

    if not os.path.exists(filepath):
        print(f"Spectrum file not found: {filepath}")
        return None, None

    Er_val, dN_dEr_val = _read(filepath)
    scale = (sigma_f / sigma_i).decompose().value
    return Er_val * Er_unit, scale * dN_dEr_val * dN_dEr_unit
