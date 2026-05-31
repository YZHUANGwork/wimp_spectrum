import numpy as np
import astropy.units as u
import astropy.constants as const
#Ad depends on v_min, v_min depends on M_T (detector material, picked), M_chi (wimp mass) , Er (selected)
#                → Ad depends on m_chi (wimp mass)


def get_SHM_Ads(Ers, m_chi, m_T,
                vc   = 220 * u.km / u.s,
                vesc = 553 * u.km / u.s,
                xp   = 0.89):
    """
    Annual modulation amplitude A_d in the SHM.
    Ref: arXiv:1112.0524 p11 Eq. 23

    Parameters
    ----------
    Ers   : recoil energies (astropy Quantity array)
    m_chi : DM mass (astropy Quantity)
    m_T   : target nucleus mass (astropy Quantity)
    vc    : local circular speed       (default: 220 km/s)
    vesc  : galactic escape speed      (default: 553 km/s)
    xp    : SHM transition parameter   (default: 0.89)

    Returns
    -------
    Ads : np.ndarray — modulation amplitudes (dimensionless)
    """
    xs = (np.sqrt(Ers) * (m_T + m_chi) / np.sqrt(m_T) / m_chi
          / (np.sqrt(2) * vc) * const.c).decompose()

    Ads       = np.zeros(len(xs))
    xs_div_xp = xs / xp
    z         = vesc / vc

    select_low = xs <= xp
    select_mid = (xs > xp) & (xs <= z)
    # xs > z: outside escape speed, Ad stays 0

    if select_low.any():
        Ads[select_low] = (0.034
                           * (xs_div_xp[select_low] - 1)
                           * (xs_div_xp[select_low] + 1))

    if select_mid.any():
        Ads[select_mid] = (0.014
                           * (xs_div_xp[select_mid] - 1)
                           * (xs_div_xp[select_mid] + 3.7))

    return Ads
