import numpy as np
import math
import random as rd
from Galaxy_model import Galaxy_model
from Set_model_parameters import Set_model_parameters as sp
from Galaxy_instruments import Galaxy_instruments

class Set_init(Galaxy_model):
    """ Sets initial distributions of coordinates and velocities. 
    Governs assymetric drift by solving Jeans equation 
    (Binney & Tremaine 2008, eq 4.222a).

    """
    
    fkv = sp.fkv
    G = sp.G
    r_min = sp.r_min
    r_max = sp.r_max
    omega_bar = sp.omega_bar
    rs = sp.rs
    dr = r_max/300
    s_R = sp.s_R
    N = sp.N
    
    def find_Q_r(self):
        """ Finds parameters of disc stability against axisymmetric 
        perturbations according to Toomre criretion.
        ---------------
        Output:
        Q: array of of 300 Toomre parameters along the Galactocentric 
        distance in the range 0.04 - 12 kpc.
        r_rep: Galactocentric distance of the minimum Toomre parameter, kpc.
        
        """
        kappa = Galaxy_model.find_kappa(self)*self.fkv
        
        Sigm_0 = sp.M_disc/(2*math.pi*sp.r_disc**2*(1 - \
               math.exp(-self.r_max/sp.r_disc)* \
                   (1 + (self.r_max/sp.r_disc))))
        Sigm = Sigm_0*np.exp(-self.rs/sp.r_disc)        
        Q_test = kappa/3.36/self.G/Sigm
        ind = np.where(Q_test == np.min(Q_test))[0][0]
        r_rep = self.rs[ind]
        
        sm_R = 3.36*self.G*Sigm[ind]/kappa[ind]
        C_R = sm_R/np.exp(-r_rep/self.s_R)
        sigm_R = C_R*np.exp(-self.rs/self.s_R)
        
        Q = sigm_R*kappa/3.36/self.G/Sigm
        return Q, r_rep
        
        
    def find_sigmas_vels(self):
        """ Returns arrays of 300 mean velocities and velocity dispersions 
        in km s^-1 along the Galactocentric distance in the range 
        0.04 - 12 kpc.     
        ---------------
        Output: 
        vt: mean azimuthal velocity.
        svr: radial velocity dispersion.
        svt: azimuthal velocity dispersion.
        """
        _, r_rep = self.find_Q_r()
        
        kappa = Galaxy_model.find_kappa(self)*self.fkv
        vc_tot = Galaxy_model.total_curve(self)
        
        omega = vc_tot/self.rs*self.fkv
        
        Qt = 1
        Sigm_0 = sp.M_disc/(2*math.pi*sp.r_disc**2*(1 - \
                math.exp(-self.r_max/sp.r_disc)* \
                    (1 + (self.r_max/sp.r_disc))))
            
        d = abs(self.rs - r_rep)
        ind = np.where(d == np.min(d))[0][0]
        
        sig_vr_0 = Qt*3.36*Sigm_0*np.exp(-r_rep/sp.r_disc)*self.G/kappa[ind]
        sr_0 = sig_vr_0*np.exp(r_rep/self.s_R)
        gamma = np.empty(300)
        sig_vr_2 = np.empty(300)
        sig_vt_2 = np.empty(300)
        svt = np.empty(300)
        svr = np.empty(300)
        vt = np.empty(300)
        vt_2 = np.empty(300)
        for i in range(0, 300):
            r = self.rs[i]
            gamma[i] = (kappa[i]/omega[i])**2/4
            sig_vr_2[i] = (sr_0*np.exp(-r/self.s_R))**2
            sig_vt_2[i] = sig_vr_2[i]*gamma[i]
            vt_2[i] = vc_tot[i]**2 + sig_vr_2[i]/self.fkv**2* \
                          (1 - gamma[i] - r/sp.r_disc - 2*r/self.s_R)
            svr[i] = np.sqrt(sig_vr_2[i])/self.fkv
            svt[i] = np.sqrt(sig_vt_2[i])/self.fkv
            vt[i] = np.sqrt(abs(vt_2[i]))
        return vt, svr, svt
    
    def set_initial(self):
        """ Returns a dictionary of initial parameters of N particles.
        j: index of the model particle.
        sts: status of the particle. 0 if the particle leaved the Galaxy 
        or fell on the central black hole.
        x, y: initial Cartesian coordinates in kpc.
        vx, vy: initial Cartesian velocities in km s^-1 calculated according
        to Binney & Tremaine 2008, eq 4.222a.
        
        """
        
        vt, svr, svt = self.find_sigmas_vels()
        
        init = {}
        
        init['j'] = np.arange(1, self.N + 1)
        init['sts'] = np.ones(self.N)
        init['x'] = np.empty(self.N)
        init['y'] = np.empty(self.N)
        init['vx'] = np.empty(self.N)
        init['vy'] = np.empty(self.N)
        
        r_init = np.empty(self.N)
        
        for j in range(self.N):
            r = 255
            while r >= self.r_max or r <= self.r_min:  
                s_1 = rd.uniform(0, 1)
                s_2 = rd.uniform(0, 1)
                r = -(math.log(s_1) + math.log(s_2))*self.r_disc
            r_init[j] = r
        r_init = np.sort(r_init)
        
        for j in range(self.N):
            r = r_init[j]
            s = rd.uniform(0, 1)
            th = math.pi*s
            x = r*math.cos(th)
            y = r*math.sin(th)
            
            init['x'][j] = x
            init['y'][j] = y

            d = abs(self.rs - r)
            ind = np.where(d == np.min(d))[0][0]
            
            vt_0 = vt[ind-1] + (vt[ind] - vt[ind-1])/(self.rs[ind] - \
                                self.rs[ind-1])*(r - self.rs[ind-1])
            sig_vr_1 = svr[ind-1] + (svr[ind] - svr[ind-1])/(self.rs[ind] - \
                                    self.rs[ind-1])*(r - self.rs[ind-1])
            sig_vt_1 = svt[ind-1] + (svt[ind] - svt[ind-1])/(self.rs[ind] - \
                                    self.rs[ind-1])*(r - self.rs[ind-1])
            vt_s = vt_0 + sig_vt_1*rd.gauss(0, 1)
            vr_s = sig_vr_1 * rd.gauss(0, 1)
            vx, vy = Galaxy_instruments.vel_gal_to_cart(self, vr_s, vt_s, x, y)
            init['vx'][j] = vx
            init['vy'][j] = vy
        return init



    