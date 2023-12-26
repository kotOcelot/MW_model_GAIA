import numpy as np

class Set_model_parameters():
    """ fkv: factor to convert kpc/Myr to km/s.
    G: Gravitational constant in kpc**3*10^-11 Msun*Myr^-2.
    N: number of model particles.
    r_min: Galactocentric distance of the last stable orbit. The particle
    falls on central black hole and integration stopped if r < r_min. Kpc.
    r_max: Galactocentric distance of the last stable orbit. The particle
    leaves the Galaxy and integration stopped if r > r_max. Kpc.
    rs: array of galactocentric coordinates used in modelling, kpc.
    dr: step of modelling, kpc.
    omega_bar: angular velocity of the bar rotation, km/s/kpc
    time: time from the start of the integration, Myr.
    M_bar: mass of the bar, M_sun.
    a_bar, b_bar: major and minor semi-axes of the bar, kpc. 
    N_rot: number of the rotations during which bar is gradually turning on.
    M_buldge: mass of the buldge, M_sun.
    r_buldge: radius of the buldge, kpc.
    M_disc: mass of the disc, M_sun.
    r_disc: exponential scale of the disc, kpc.
    s_R: observed radial scale of the velocity dispersion, kpc.
    v_max: maximum velocity of the halo, km/s.
    r_halo: radius of the halo, kpc.

    """
    
    fkv = 3.1556925/3.0856776/1000
    G = 6.67408*1.98892/(3.0856776**2)*(3.1556925**2)/3.0856776*0.1 
    N = 2000000
    
    r_min = 0.02 
    r_max = 11
    rs = np.arange(12/300, 12  +  12/300, 12/300)
    dr = r_max/300
    
    omega_bar = 55
    time = 1500
    
    M_bar = 0.13
    a_bar = 4.2
    b_bar = 1.35
    N_rot = 4 
    
    M_buldge = 0.05
    r_buldge = 0.3
    
    M_disc = 0.325
    r_disc = 2.5
    s_R = 20
    
    v_max = 201.4*fkv
    r_halo = 8  
    
    
    
    