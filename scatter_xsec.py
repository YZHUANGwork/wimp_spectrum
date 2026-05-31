import numpy as np
import os

import random
import math
import matplotlib.pyplot as plt
import string
import astropy.units as u
import astropy.constants as const
from scipy.special import spherical_jn, erf

class xsecEr_WIMPN_el():
    def __init__(self, A, m_chi, Er, 
                 unit_convert = (const.hbar*const.c).to(u.keV*u.fm),
                 xsec_WIMPnucleon = 1e-45* u.cm**2, s = 0.9*u.fm, r_0 = 0.52 *u.fm
                 ):
        """
        
        A: mass number
        g_vmin: halo structure term
        
        Er: WIMP-nucleus recoil energy
        """
        
        self.A = A
        self.m_chi = m_chi
        
        self.unit_convert = unit_convert
        self.xsec_WIMPnucleon = xsec_WIMPnucleon
        
        
        self.m_N = self.A * (const.m_n * const.c**2).to(u.GeV)
        self.m_p = (const.m_p * const.c**2).to(u.GeV)
        
        self.Er = Er
        self.q = np.sqrt(2 * self.m_N  * self.Er).to(u.keV)
        
        
        #helm form factor https://www.tir.tw/phys/hep/dm/amidas/equations/eq-FQ_Helm.html
        self.s = s #nuclear skin thickness
        self.r_0 = r_0
        self.R_A = (1.23* self.A**(1/3) - 0.6)*u.fm 
        self.R = np.sqrt(self.R_A**2 +(7/3) * np.pi**2 * self.r_0**2 - 5 * self.s**2) #effective nuclear radius
        
    def xsec_WIMPnucleus(self):
        self.miu_WIMPN = (self.m_chi * self.m_N)/(self.m_chi + self.m_N) # WIMP-nucleus reduced mass
        self.miu_WIMPnucleon =  (self.m_chi * self.m_p )/(self.m_chi + self.m_p ) #WIMP-nucleon reduced mass
        
        #https://arxiv.org/pdf/astro-ph/0307190 p3 eqn7
        xsec_WIMPnucleus = (self.xsec_WIMPnucleon * 
                                    (self.miu_WIMPN / self.miu_WIMPnucleon)**2 * self.A**2
                                   ).to(u.cm**2)
        return xsec_WIMPnucleus
    
    
    def F2_Q(self):
        # first Born (plane wave) approximation
        R_nucleus = self.R
        x = self.q * R_nucleus
        
        x_dim =  (x/ self.unit_convert ).decompose()
        if x_dim.unit !=(u.m/u.m):
            print('dimension error')
            return 0
        exponential_term = x*self.s/R_nucleus
        exponential_term_dim = (exponential_term / self.unit_convert).decompose()
        F2_q = (3 * spherical_jn(1, x_dim.value) / (x_dim.value))**2 * np.exp(-(exponential_term_dim.value)**2) 

        return F2_q
    

    def xsec_Er(self):
        #eqn2 p3 https://arxiv.org/pdf/astro-ph/0307190.pdf
        #eqn3 p4 https://arxiv.org/abs/1209.3339.pdf

        sigma_0 = self.xsec_WIMPnucleus()#self.get_effective_total_crosssection()
        F2_q = self.F2_Q()
        xsec_Er = sigma_0 * F2_q
        return xsec_Er 