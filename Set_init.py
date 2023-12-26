import numpy as np
import math
from Galaxy_instruments import Galaxy_instruments
from Galaxy_model import Galaxy_model

class Set_init(Galaxy_model):
    """
    Sets initial distributions of coordinates and velocities. 
    Governs assymetric drift by solving Jeans equation 
    (Binney & Tremaine 2008, eq 4.222a).
    Parameters:
    fkv: factor to convert kpc/Myr to km/s.
    G: Gravitational constant in kpc**3*10^-11 Msun*Myr^-2.
    r_min: Galactocentric distance of the last stable orbit. The particle
    falls on central black hole and integration stopped if r < r_min. Kpc.
    r_max: Galactocentric distance of the last stable orbit. The particle
    leaves the Galaxy and integration stopped if r > r_max. Kpc.
    omega_bar: angular velocity of the bar rotation, km/s/kpc. Default: 55.
    rs: array of galactocentric coordinates used in modelling, kpc.
    dr: step of modelling, kpc.
    s_R: radial scale of the velocity dispersion, kpc.
    """
    
    fkv = 3.1556925/3.0856776/1000
    G = 6.67408*1.98892/(3.0856776**2)*(3.1556925**2)/3.0856776*0.1 
    r_min = 0.02 #kpc
    r_max = 11 #kpc
    omega_bar = 55 #km*s^-1*kpc^-1
    rs = np.arange(12/300, 12  +  12/300, 12/300) #kpc
    dr = r_max/300
    s_R = 20

    def find_Q_r(self):
        """ Finds parameters of disc stability against axisymmetric 
        perturbations according to Toomre criretion.
        Output: Q: array of of 300 Toomre parameters along the Galactocentric 
        distance in the range 0.04 - 12 kpc.
        r_rep: Galactocentric distance of the minimum Toomre parameter, kpc.
        
        """
        
        kappa = self.find_kappa()*self.fkv
        
        Sigm_0 = self.M_disc/(2*math.pi*self.r_disc**2*(1 - \
               math.exp(-self.r_max/self.r_disc)* \
                   (1 + (self.r_max/self.r_disc))))
        Sigm = Sigm_0*np.exp(-self.rs/self.r_disc)        
        Q_test = kappa/3.36/self.G/Sigm
        ind = np.where(Q_test == np.min(Q_test))[0][0]
        r_rep = self.rs[ind]
        
        s_R = 3.36*self.G*Sigm[ind]/(kappa[ind])
        C_R = -s_R/np.exp(-r_rep/self.s_R)
        sigm_R = C_R*np.exp(-self.rs/self.s_R)
        
        
        Q = sigm_R*kappa/3.36/self.G/Sigm
        return Q, r_rep
        
        
    def find_sigmas_vels(self):
        """ Returns arrays of 300 mean velocities and velocity dispersions 
        in km s^-1 along the Galactocentric distance in the range 
        0.04 - 12 kpc.     
        (Binney & Tremaine 2008, eq 4.222a).
        Output: vt: mean azimuthal velocity.
        svr: radial velocity dispersion.
        svt: aximuthal velocity dispersion.
        """
        _, r_rep = self.find_Q_r()
        rs = self.rs
        kappa = self.find_kappa()*self.fkv
        vc_tot = self.total_curve()
        omega = vc_tot/self.rs*self.fkv
        Qt = 1
        Sigm_0 = self.M_disc/(2*math.pi*self.r_disc**2*(1 - \
               math.exp(-self.r_max/self.r_disc)* \
                   (1 + (self.r_max/self.r_disc))))
            
        for k in range(1, 301):
            if (r_rep > rs[k-1]) and (r_rep <= rs[k]):
                ind = k
        sig_vr_0 = Qt*3.36*Sigm_0*np.exp(-r_rep/self.r_disc)*self.G/kappa[ind]
        sr_0 = sig_vr_0*np.exp(r_rep/self.s_R)
        gamma = np.empty(300)
        sig_vr_2 = np.empty(300)
        sig_vt_2 = np.empty(300)
        svt = np.empty(300)
        svr = np.empty(300)
        vt = np.empty(300)
        vt_2 = np.empty(300)
        for i in range(0, 300):
            r = rs[i]
            gamma[i] = (kappa[i]/omega[i])**2/4
            sig_vr_2[i] = (sr_0*np.exp(-r/self.s_R))**2
            sig_vt_2[i] = sig_vr_2[i]*gamma[i]
            vt_2[i] = vc_tot[i]**2 + sig_vr_2[i]/self.fkv**2* \
                          (1 - gamma[i] - r/self.r_disc - 2*r/self.s_R)
            svr[i] = np.sqrt(sig_vr_2[i])/self.fkv
            svt[i] = np.sqrt(sig_vt_2[i])/self.fkv
            vt[i] = np.sqrt(abs(vt_2[i]))
        return vt, svr, svt

    