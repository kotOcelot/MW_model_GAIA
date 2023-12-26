import numpy as np
from Set_model_parameters import Set_model_parameters as sp

class Galaxy_instruments():
    """
    Basic transforms used both in Galaxy modeliing and processing.

    """
    
    fkv = sp.fkv
    omega_bar = sp.omega_bar 
        
    def cart_to_pol(self, x, y):
        """Converts the Cartesian coordinates x, y to the polar coordinates 
        r and phi. 
        ---------------
        phi: radians in the range [0, 2*pi]

        """

        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y,x) % 2*np.pi
        return r, phi

        
    def pol_to_cart(self, r, phi):
        """Converts the polar coordinates r, phi to the Cartesian coordinates 
        x and y. 
        ---------------        
        phi: radians.

        """
    
        x = r * np.cos(phi)
        y = r * np.sin(phi)    
        return x, y
    
    def cart_to_bar(self, x, y):
        """Converts the Cartesian coordinates x, y to the Cartesian 
        coordinates xf and yf in the bar reference system. 
        
        """
        
        theta = self.time*self.omega_bar*self.fkv
        
        xf = x*np.cos(theta) + y*np.sin(theta)
        yf = -x*np.sin(theta) + y*np.cos(theta)
        return xf, yf
    
    def bar_to_cart(self, xf, yf):
        """Converts the Cartesian coordinates xf, yf in the bar reference 
        system to the Cartesian coordinates x and y. 

        """
        
        theta = self.time*self.omega_bar*self.fkv

        x = xf*np.cos(theta) - yf*np.sin(theta)
        y = xf*np.sin(theta) + yf*np.cos(theta)
        return x, y
    
    def vel_gal_to_cart(self, vr, vt, x, y):
        """Converts the Galactocentric velocities vr, vt to Cartesian 
        velocities vx, vy. 
        ---------------        
        x, y: Cartesian coordinates. 
        
        """
        r, phi = self.cart_to_pol(x, y)
        vx = vr*np.cos(phi) - vt*np.sin(phi)
        vy = vr*np.sin(phi) + vt*np.cos(phi)
        return vx, vy

    def vel_cart_to_gal(self, vx, vy, x, y):
        """Converts the Cartesian velocities vx, vy to Galactocentric 
        velocities vr, vt. 
        
        
        """
        r, phi = self.cart_to_pol(x, y)
        vr = vx*np.cos(phi) + vy*np.sin(phi)
        vt = -vx*np.sin(phi) + vy*np.cos(phi)
        return vr, vt
    