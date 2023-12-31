import numpy as np
import math
from Galaxy_instruments import Galaxy_instruments
from Set_model_parameters import Set_model_parameters as sp
from scipy import special as spf
import matplotlib.pyplot as plt

class Galaxy_model(Galaxy_instruments):
    """ Components of the Galaxy potential, rotation curve and resonances.
    
    """
    
    def __init__(self):
        """ Initializing default parameters.
        
        """
        self.fkv = sp.fkv
        self.G = sp.G 
        
        self.r_min = sp.r_min
        self.r_max = sp.r_max
        self.rs = sp.rs
        self.dr = self.r_max/300

        self.omega_bar = sp.omega_bar
        
        self.time = sp.time
        
        self.M_bar = sp.M_bar
        self.a_bar = sp.a_bar
        self.b_bar = sp.b_bar
        
        self.M_buldge = sp.M_buldge
        self.r_buldge = sp.r_buldge
        
        self.M_disc = sp.M_disc
        self.r_disc = sp.r_disc
        
        self.v_max = sp.v_max
        self.r_halo = sp.r_halo
        
        self.N_rot = sp.N_rot
        
        self.vc_bar, self.acc_bar = self.bar_curve()
        self.vc_disc, self.acc_disc = self.disc_curve()
        
    def accel_bar(self, x, y, return_potential=False):
        """ Returns the acceleration produced by the Ferrer's bar, n = 2 
        (Binney & Tremaine 2008, p.95) in given point.
        Args: 
        x, y: Cartesian coordinates.
        return_potential: if True, returns value of the bar gravitational 
        potential in given point.
        ---------------
        Output:
        ax, ay: Cartesian components of acceleration by bar in given point.
        
        """
        
        r = math.sqrt(x**2 + y**2)
        
        if r > (self.r_max + 0.002):
            print('Error! Particle leaves the Galaxy!')
            return
        if r < self.r_min:
            print('Error! Particle falls on center!')
            return
            
        eps = math.sqrt(self.a_bar**2 - self.b_bar**2)
        c_bar = 105/32*self.M_bar*self.G/eps
        x_bar, y_bar = Galaxy_instruments.cart_to_bar(self, x, y)
            
        mu = (x_bar/self.a_bar)**2 + (y_bar/self.b_bar)**2
        ints = {}
        #de Vaucouleurs & Freeman 1972, Apendix, Ferres's bar n=2
        if mu < 1:
            ints['1'] = math.log((self.a_bar + eps)/self.b_bar)
            ints['2'] = 0.5*eps*self.a_bar/self.b_bar**2 + 0.5*ints['1']
            ints['3'] = 0.25*eps*self.a_bar**3/self.b_bar**4 + 3/4*ints['2']
            ints['4'] = 1/6*eps*self.a_bar**5/self.b_bar**6 + 5/6*ints['3']
            ints['5'] = 1/3*eps*self.b_bar**2/self.a_bar**3 + 2/3*eps/self.a_bar
            ints['6'] = 1/5*eps*self.b_bar**4/self.a_bar**5 + 4/5*ints['5']
            cos_p = self.b_bar/self.a_bar
            sin_p = eps/self.a_bar
        else:
            d = (eps**2 + y_bar**2 - x_bar**2)**2 + 4*(x_bar*y_bar)**2
            if abs(x_bar) > 0.01:
                cos_p = math.sqrt(-eps**2 - y_bar**2 + x_bar**2 \
                        + math.sqrt(d))/math.sqrt(2)/abs(x_bar) 
            else:
                cos_p = abs(y_bar)/math.sqrt(eps**2 + y_bar**2)
            sin_p = 1.0
            if cos_p < 1.0: 
                sin_p = math.sqrt(1 - cos_p**2)

            ints['1'] = math.log(abs((1 + sin_p)/cos_p))
            ints['2'] = 0.5*sin_p/cos_p**2 + 0.5*ints['1']
            ints['3'] = 0.25*sin_p/cos_p**4 + 3/4*ints['2']
            ints['4'] = 1/6*sin_p/cos_p**6 + 5/6*ints['3']
            ints['5'] = 1/3*sin_p*cos_p**2 + 2/3*sin_p
            ints['6'] = 1/5*sin_p*cos_p**4 + 4/5*ints['5']
            
        w = {}
        w['10'] = 2*ints['1']
        w['11'] = (2*ints['1'] - 2*sin_p)/eps**2
        w['20'] = (2*ints['2'] - 2*ints['1'])/eps**2
        w['12'] = (2*ints['1'] - 4*sin_p + 2*ints['5'])/eps**4
        w['21'] = (2*ints['2'] - 4*ints['1'] + 2*sin_p)/eps**4
        w['30'] = (2*ints['3'] - 4*ints['2'] + 2*ints['1'])/eps**4
        w['13'] = (2*ints['1'] - 6*sin_p + 6*ints['5'] - 2*ints['6']) \
                   /3/eps**6
        w['22'] = (2*ints['2'] - 6*ints['1'] + 6*sin_p - 2*ints['5']) \
                   /3/eps**6
        w['31'] = (2*ints['3'] - 6*ints['2'] + 6*ints['1'] - 2*sin_p) \
                   /3/eps**6
        w['40'] = (2*ints['4'] - 6*ints['3'] + 6*ints['2'] - 2*ints['1']) \
                   /3/eps**6
                   
        pot_bar = -c_bar*(1/3*w['10'] - w['11']*x_bar**2 - \
                  w['20']*y_bar**2 + w['12']*x_bar**4 + \
                  2*w['21']*(x_bar*y_bar)**2 + w['30']*y_bar**4 - \
                  w['13']*x_bar**6 - 3*w['22']*(y_bar*x_bar**2)**2 - \
                  3*w['31']*(x_bar*y_bar**2)**2 - w['40']*y_bar**6)
            
        ax_bar = c_bar*(-2*w['11']*x_bar + 4*w['12']*x_bar**3 + \
                 4*w['21']*x_bar*y_bar**2 - 6*w['13']*x_bar**5 - \
                 12*w['22']*x_bar*(x_bar*y_bar)**2 - \
                 6*w['31']*x_bar*y_bar**4)
        ay_bar = c_bar*(-2*w['20']*y_bar + 4*w['21']*y_bar*x_bar**2 + \
                 4*w['30']*y_bar**3 - 6*w['22']*y_bar*x_bar**4 - \
                 12*w['31']*y_bar*(x_bar*y_bar)**2 - 6*w['40']*y_bar**5)
            
        ax, ay = Galaxy_instruments.bar_to_cart(self, ax_bar, ay_bar)
        if return_potential:
            return ax, ay, pot_bar
        else:
            return ax, ay
        
    def bar_curve(self):
        """ Returns array of 300 mean velocities in km s^-1 produced 
        by the potential of the bar along the Galactocentric distance 
        in the range 0.04 - 12 kpc.
        
        """
        
        n_int = 1000
        d_fi = 2*math.pi/n_int
        
        vc_bar = np.zeros(300)
        acc_bar = np.zeros(300)
        
        for i in range(1, 301):
            r = i*self.dr
            a_sum = 0
            for j in range (1, 1 + n_int): #Mean on 360 degrees
                fi = (j - 0.5)*d_fi
                x = r*math.cos(fi)
                y = r*math.sin(fi)
                ax, ay = self.accel_bar(x, y)
                a_sum += ax*math.cos(fi) + ay*math.sin(fi)
                
            vc_bar[i-1] = math.sqrt(r*abs(a_sum)*d_fi/2/math.pi)/self.fkv
            acc_bar[i-1] = a_sum*d_fi/2/math.pi
        return vc_bar, acc_bar
    
    def bar_turn_on(self, x, y, return_potential=False):
        """ Returns the acceleration produced by the bar gradually turning on 
        during N_rot rotatoins in given point.
        Args:
        x, y: Cartesian coordinates.
        return_potential: if True, returns value of the bar gravitational 
        potential in given point.
        ---------------
        Output:
        ax, ay: Cartesian components of acceleration by bar in given point.
        
        """
        
        tbrot = 2*math.pi/self.omega_bar/self.fkv
        tgrow = self.N_rot*tbrot
         
        if self.time < tgrow:
            bar_str = self.time/tgrow
        else:
            bar_str = 1.0
            
        ax, ay = self.accel_bar(x, y)
        
        r = np.sqrt(x**2 + y**2)
        
        if bar_str < 1:
            d = abs(self.rs - r)
            ind = np.where(d == np.min(d))[0][0]
            vcb = self.vc_bar[ind-1] + (self.vc_bar[ind] - self.vc_bar[ind-1]) * \
                  (r - self.rs[ind-1])/(self.rs[ind] - self.rs[ind-1])
            ax = bar_str*ax - (1 - bar_str)*vcb**2*x/r**2*self.fkv**2
            ay = bar_str*ay - (1 - bar_str)*vcb**2*y/r**2*self.fkv**2
        return ax, ay
    
    def accel_buldge(self, x, y, return_potential=False):
        """ Returns the acceleration produced by the buldge in given point.
        Args:
        x, y: Cartesian coordinates.
        return_potential: if True, returns value of the bar gravitational 
        potential in given point.
        ---------------
        Output:
        ax, ay: Cartesian components of acceleration by buldge in given point.
        
        """
  
        r = math.sqrt(x**2 + y**2)
        
        if r > (self.r_max + 0.002):
            print('Error! Particle leaves the Galaxy!')
            return
        if r < self.r_min:
            print('Error! Particle falls on center!')
            return

        pot_buldge = -self.G*self.M_buldge/math.sqrt(r**2 + self.r_buldge**2)
        a_buldge = -self.G*self.M_buldge*r/((r**2 + self.r_buldge**2)* \
                   math.sqrt(r**2 + self.r_buldge**2))
                   
        ax = a_buldge*x/r
        ay = a_buldge*y/r      
                 
        if return_potential:
            return ax, ay, pot_buldge
        else:
            return ax, ay
        
    def buldge_curve(self):
        """ Returns array of 300 mean velocities in km s^-1 produced by the 
        potential of the buldge along the Galactocentric distance 
        in the range 0.04 - 12 kpc.
        
        """
        
        vc_buldge = np.empty(300)
        
        for i in range(1, 301):
            r = i*self.dr
            a_sum = 0
            x = r
            y = 0
            ax, ay = self.accel_buldge(x, y)
            a_sum = np.sqrt(ax**2 + ay**2)
            vc_buldge[i-1] = math.sqrt(r*abs(a_sum))/self.fkv
        return vc_buldge
                      
    def accel_disc(self, x, y, return_potential=False):
        """ Returns the acceleration produced by the disc in given point.
        Args:
        x, y: Cartesian coordinates.
        ---------------
        Output:
        ax, ay: Cartesian components of acceleration by disc in given point.
        
        """
        
        r = math.sqrt(x**2 + y**2)
        
        if r > (self.r_max + 0.002):
            print('Error! Particle leaves the Galaxy!')
            return
        if r < self.r_min:
            print('Error! Particle falls on center!')
            return

        d = abs(self.rs - r)
        ind = np.where(d == np.min(d))[0][0]
                
        if r < self.rs[0] and r >= self.r_min:
            ind = 0
                
        acc_disc_loc = self.acc_disc[ind] + \
                     (self.acc_disc[ind+1] - self.acc_disc[ind]) * \
                     (r - self.rs[ind])/(self.rs[ind+1] - \
                                           self.rs[ind])
        ax = acc_disc_loc*x/r
        ay = acc_disc_loc*y/r
        return ax, ay
    
    def disc_curve(self):
        """ Returns arrays of 300 mean velocities in km s^-1 and 
        accelerations produced by the potential of the disc along the 
        Galactocentric distance in the range 0.04 - 12 kpc.
        
        """
        
        vc_disc = np.empty(300)
        acc_disc = np.empty(300)
        
        for i in range(1, 301):
            r = i*self.dr
            y_d = r/2/self.r_disc
            sigm = self.M_disc/(2*math.pi*self.r_disc**2*(1 - \
                   math.exp(-self.r_max/self.r_disc)* \
                       (1 + (self.r_max/self.r_disc))))
            acc_disc[i-1] = -(4*math.pi*self.G*sigm*self.r_disc*y_d*y_d* \
                            (spf.iv(0, y_d)*spf.kv(0, y_d) - \
                             spf.iv(1, y_d)*spf.kv(1, y_d)))/r
            vc_disc[i-1] = math.sqrt(abs(r*acc_disc[i-1]))/self.fkv
        return vc_disc, acc_disc
    

    def accel_halo(self, x, y, return_potential=False):
        """ Returns the acceleration produced by the halo in given point.
        Args:
        x, y: Cartesian coordinates.
        return_potential: if True, returns value of the bar gravitational 
        potential in given point.
        ---------------
        Output:
        ax, ay: Cartesian components of acceleration by disc in given point.
        
        """
  
        r = math.sqrt(x**2 + y**2)
        
        if r > (self.r_max + 0.002):
            print('Error! Particle leaves the Galaxy!')
            return
        if r < self.r_min:
            print('Error! Particle falls on center!')
            return

        pot_hl = self.v_max**2/2*math.log((r**2 + self.r_halo**2)/ \
                                          self.r_halo**2)
        acc_hl = -self.v_max**2*r/(r**2 + self.r_halo**2)

        ax = acc_hl*x/r
        ay = acc_hl*y/r   
                    
        if return_potential:
            return ax, ay, pot_hl
        else:
            return ax, ay
    
    def halo_curve(self):
        """ Returns array of 300 mean velocities in km s^-1 produced by
        the potential of the halo along the Galactocentric distance 
        in the range 0.04 - 12 kpc.
        
        """  
        
        vc_halo = np.zeros(300)
        
        for i in range(1, 301):
            r = i*self.dr
            x = r
            y = 0
            ax, ay = self.accel_halo(x, y)
            a_sum = np.sqrt(ax**2 + ay**2)
            vc_halo[i-1] = math.sqrt(r*abs(a_sum))/self.fkv
        return vc_halo
    
    def accel_total(self, x, y, time):
        """ Returns the acceleration produced by the total Galaxy potential
        in given point.
        Args:
        x, y: Cartesian coordinates.
        time: time from the start of the integration, Myr.
        ---------------
        Output:
        ax, ay: Cartesian components of acceleration by disc in given point.
        
        """
        self.time = time
        
        ax_bar, ay_bar = self.bar_turn_on(x, y)
        ax_buldge, ay_buldge = self.accel_buldge(x, y)
        ax_disc, ay_disc = self.accel_disc(x, y)
        ax_halo, ay_halo = self.accel_halo(x, y)
        
        ax = ax_bar + ax_buldge + ax_disc + ax_halo
        ay = ay_bar + ay_buldge + ay_disc + ay_halo
        return ax, ay
    
    def total_curve(self):
        """ Returns array of 300 mean velocities in km s^-1 produced by
        the total potential of the Galaxy along the Galactocentric 
        distance in the range 0.04 - 12 kpc.
        
        """    
        
        vc_tot = np.empty(300)
        
        for i in range(1, 301):
            r = i*self.dr
            x = r
            y = 0
            acc_bar = abs(self.acc_bar[i-1])
            ax_buldge, ay_buldge = self.accel_buldge(x, y)
            acc_buldge = np.sqrt(ax_buldge**2 + ay_buldge**2)
            acc_disc = abs(self.acc_disc[i-1])
            ax_halo, ay_halo = self.accel_halo(x, y)
            acc_halo = np.sqrt(ax_halo**2 + ay_halo**2)
            a_sum = acc_bar + acc_buldge + acc_disc + acc_halo
            vc_tot[i-1] = math.sqrt(r*abs(a_sum))/self.fkv
        return vc_tot
    
    def find_omega(self):
        """ Returns array of 300 mean angular velocities in km s^-1 kpc^-1
        along the Galactocentric distance in the range 0.04 - 12 kpc.
        
        """
        omega = self.total_curve()/self.rs
        return omega
    
    def find_kappa(self):
        """ Returns array of 300 epicycle frequences in km s^-1 kpc^-1
        along the Galactocentric distance in the range 0.04 - 12 kpc.
        (Binney & Tremaine 2008, eq 3.80)
        
        """
        
        kappa = np.empty(300)
        vc_tot = self.total_curve()
        omega = vc_tot/self.rs
        
        der_1 = (2*(omega[1] - omega[0]) - 1/2*(omega[2] - \
                                                 omega[0]))/self.rs[0]
        kappa[0] = 2*omega[0]*math.sqrt(1 + 
                                        self.rs[0]/2/omega[0]*der_1)
        
        der_1 = (2*(omega[2] - omega[1]) - 
                 1/2*(omega[3] - omega[1]))/self.rs[1]
        kappa[1] = 2*omega[1]*math.sqrt(1 + self.rs[1]/2/omega[1]*der_1)
        
        for i in range (2, 298):
            y0 = vc_tot[i]/self.rs[i]
            yp1 = vc_tot[i+1]/(self.rs[i] + self.dr)
            yp2 = vc_tot[i+2]/(self.rs[i] + 2*self.dr)
            ym1 = vc_tot[i-1]/(self.rs[i] - self.dr)
            ym2 = vc_tot[i-2]/(self.rs[i] - 2*self.dr)
            der_1 = ((yp1 - y0) - (ym1 - y0) - 1/4*(yp2 - y0) + \
                     1/4*(ym2 - y0))/self.dr
            kappa[i] = 2*omega[i]* \
                       math.sqrt(abs(1 + self.rs[i]/2/omega[i]*der_1))
            
        s = self.rs[297]/2/omega[297]*der_1
        kappa[298:] = 2*omega[298:]*math.sqrt(1 + s)
        return kappa
    
    def find_CR(self):
        """ Returns Galactocentric distance of the corotation 
        (1:1 resonance) in kpc.
        
        """
        
        omega = self.find_omega()
        
        d = abs(self.omega_bar - omega)
        ind = np.where(d == np.min(d))[0][0]
        
        r_cr = self.rs[ind-1] + (self.rs[ind] - self.rs[ind-1])/ \
             (omega[ind] - omega[ind-1])*(self.omega_bar - omega[ind-1])
        return r_cr
     
    def find_OLR(self):
        """ Returns Galactocentric distance of the OLR (-2:1 resonance) in kpc.
        
        """
        
        omega = self.find_omega()
        kappa = self.find_kappa()
        
        d = abs(self.omega_bar - omega - kappa/2)
        ind = np.where(d == np.min(d))[0][0]

        r_olr = self.rs[ind-1] + (self.rs[ind] - self.rs[ind-1])/ \
             (omega[ind] + kappa[ind]/2 - omega[ind-1] - kappa[ind-1]/2) * \
             (self.omega_bar - omega[ind-1] - kappa[ind-1]/2)
        return r_olr        
    
    def find_41(self):
        """ Returns Galactocentric distance of the -4:1 resonance in kpc.
        
        """
        
        omega = self.find_omega()
        kappa = self.find_kappa()        

        d = abs(self.omega_bar - omega - kappa/4)
        ind = np.where(d == np.min(d))[0][0]

        r_41 = self.rs[ind-1] + (self.rs[ind] - self.rs[ind-1])/ \
             (omega[ind] + kappa[ind]/4 - omega[ind-1] - kappa[ind-1]/4)* \
             (self.omega_bar - omega[ind-1] - kappa[ind-1]/4)
        return r_41   
    
                           

                
            
                                
            
            
            
            
            