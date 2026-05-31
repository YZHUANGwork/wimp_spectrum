import numpy as np
import os
import astropy.units as u
import astropy.constants as const

def get_target_dNT_dM(A, abundance = 1.):
    #p14 https://arxiv.org/pdf/2104.12785
    M_u = 1*u.g/u.mol 
    
    dN_T_dM = const.N_A/M_u/A*abundance
    return dN_T_dM

def get_target_info(target):
    if target =='Xe':
        A = 131.293
        Z = 54
        
    elif target =='Ar':
        A = 39.948
        Z = 18
        
    elif target =='He':
        A = 4       
        Z = 2
    return A, Z