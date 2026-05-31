"""
DM_halo.py
----------
Standard Halo Model (SHM) velocity distribution and derived quantities.

References
----------
- arXiv:1209.3339  (v_min, elastic/inelastic)
- arXiv:1906.10466 (v_min)
- arXiv:2104.12785 (mean inverse speed g(v_min))
"""

import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy import integrate
from scipy.special import erf


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

def get_vmin(m_N, m_chi, Er, E_excitation, print_check=False):
    """
    Minimum DM speed to produce nuclear recoil energy Er.

    Works for elastic (E_excitation = 0) and inelastic (E_excitation > 0).

    Parameters
    ----------
    m_N          : nucleus mass (astropy Quantity, energy units)
    m_chi        : DM mass     (astropy Quantity, energy units)
    Er           : recoil energy (astropy Quantity)
    E_excitation : nuclear excitation energy (astropy Quantity; 0 for elastic)
    print_check  : print debug info if True

    Returns
    -------
    v_min : astropy Quantity [km/s]

    References: arXiv:1209.3339 p5 Eq.5 / arXiv:1906.10466 p6 Eq.3.2
    """
    q   = np.sqrt(2 * m_N * Er)
    mu  = m_chi * m_N / (m_chi + m_N)  # WIMP-nucleus reduced mass

    v_min = (q / (2 * mu)).decompose() * const.c \
          + (E_excitation / q).decompose() * const.c

    if print_check:
        print(f"Er = {Er},  v_min = {v_min.to(u.km/u.s)},  "
              f"mu = {mu},  E_excitation = {E_excitation}")

    return v_min.to(u.km / u.s)


# ---------------------------------------------------------------------------
# Velocity distribution — SHM in the Earth frame
# ---------------------------------------------------------------------------

def f_v_costheta(costheta, v, v_esc, v_mag_Earth_t, sigma_v):
    """
    SHM speed distribution in the Earth frame, f(v, cos θ).

    All speed arguments must be in the same units (plain floats after
    stripping astropy units).

    Parameters
    ----------
    costheta      : cos of angle between DM velocity and Earth velocity
    v             : DM speed in Earth frame
    v_esc         : galactic escape speed
    v_mag_Earth_t : magnitude of Earth's velocity (galactic rest frame)
    sigma_v       : velocity dispersion

    Returns
    -------
    f : float  [1/v_unit^3]
    """
    v0    = np.sqrt(2 / 3) * sigma_v
    z     = v_esc / v0
    N_esc = erf(z) - 2 / np.sqrt(np.pi) * z * np.exp(-z**2)

    v_DM_gal = np.sqrt(v**2 + v_mag_Earth_t**2 + 2 * v * v_mag_Earth_t * costheta)

    return (1 / N_esc
            * (1 / np.pi / v0**2)**1.5
            * np.exp(-v_DM_gal**2 / v0**2)
            * np.heaviside(v_esc - v_DM_gal, 0))


def _g_integrand(costheta, v, v_esc, v_mag_Earth_t, sigma_v):
    """Integrand for g(v_min): f(v, cosθ) · v² / v."""
    return f_v_costheta(costheta, v, v_esc, v_mag_Earth_t, sigma_v) * v


def get_gvmin(vmin, v_mag_Earth_t,
              v_esc=553 * u.km / u.s,
              sigma_v=270 * u.km / u.s,
              v_unit=u.km / u.s):
    """
    Mean inverse speed g(v_min) = ∫_{v_min} f(v)/v d³v.

    Splits the velocity integral into the appropriate sub-domains
    depending on how v_min compares to v_esc ± v_E.

    Parameters
    ----------
    vmin          : minimum speed (astropy Quantity)
    v_mag_Earth_t : Earth speed in galactic frame (astropy Quantity)
    v_esc         : galactic escape speed (default 553 km/s)
    sigma_v       : velocity dispersion   (default 270 km/s)
    v_unit        : unit for numerical integration (default km/s)

    Returns
    -------
    g_vmin       : astropy Quantity  [1/v_unit]
    g_v_unit     : unit of the result

    Reference: arXiv:2104.12785 p128
    """
    # --- unit bookkeeping ---
    f_v_unit     = (1 / v_unit**2)**1.5
    g_v_unit     = f_v_unit * v_unit**2   # = 1/v_unit

    # --- strip units for scipy ---
    vmin_val = vmin.to(v_unit).value
    vesc_val = v_esc.to(v_unit).value
    vE_val   = v_mag_Earth_t.to(v_unit).value
    sv_val   = sigma_v.to(v_unit).value
    args     = (vesc_val, vE_val, sv_val)

    vesc_minus_vE = vesc_val - vE_val
    vesc_plus_vE  = vesc_val + vE_val

    def cos_upper(v):
        return (vesc_val**2 - v**2 - vE_val**2) / (2 * v * vE_val)

    if 0 <= vmin_val <= vesc_minus_vE:
        i1, _ = integrate.dblquad(_g_integrand,
                                   vmin_val, vesc_minus_vE,
                                   -1, 1, args=args)
        i2, _ = integrate.dblquad(_g_integrand,
                                   vesc_minus_vE, vesc_plus_vE,
                                   -1, cos_upper, args=args)
        itgl_dvdcos = i1 + i2

    elif vesc_minus_vE <= vmin_val <= vesc_plus_vE:
        itgl_dvdcos, _ = integrate.dblquad(_g_integrand,
                                            vmin_val, vesc_plus_vE,
                                            -1, cos_upper, args=args)

    else:   # vmin > v_esc + v_E — no DM reaches the detector
        itgl_dvdcos = 0.0

    g_vmin = itgl_dvdcos * 2 * np.pi * g_v_unit
    return g_vmin, g_v_unit
