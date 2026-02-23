"""
Kigali-style parametric didgeridoo shape.

### **1\. The Genetic Foundation**

First, the model takes the genome (a list of values between 0 and 1\) and scales them to real-world dimensions.

* **Length & Bell:** Determine the bounding box.  
* **Power-Law Curve:** This creates the "ideal" taper. If the power is $1.0$, it’s a straight cone; if it’s $\>1.0$, it becomes a parabolic horn.

### **2\. The Jitter Layer**

The genome provides x\_genome and y\_genome offsets. These act like "noise" that pushes the diameter in or out at various segments. This gives the didgeridoo its unique, organic character and complex overtones.

### **3\. The "Exact" Constraint Layer (RBF)**

This is where your forced\_diameters come in. To make the diameter exact at a specific $x$ without creating a sharp, unplayable edge, we use **Radial Basis Function (RBF) Interpolation**.

* **Error Calculation:** The code looks at the current diameter at your target distance ($x=800$) and calculates the "error" ($\\Delta$).  
* **Anchor Points:** We set the error at the mouthpiece ($x=0$) and the bell ($x=L$) to zero so the ends don't move.  
* **Smooth Warping:** The RBF creates a continuous mathematical function that passes **exactly** through your required $\\Delta$ at $x=800$ and $x=1400$, while remaining $0$ at the ends.  
* **Addition:** This function is added to the bore. The entire shape "bends" just enough to hit your exact numbers while preserving the "genetic" ripples from step 2\.

### **4\. Acoustic Features**

* **Bell Accent:** If a bell accent is defined, the last section of the bore is mathematically flared outward to increase volume and projection.  
* **Bubbles:** Finally, "bubbles" (sinusoidal bulges) are added. These are local expansions used to tune specific resonances or "toots" without changing the overall taper of the instrument.
"""

import numpy as np
from typing import Any, List, Tuple, Optional

from ..geo import Geo
from ..evo.genome import GeoGenome

from scipy.interpolate import Rbf # Ensure scipy is available



class KigaliShape(GeoGenome):
    """
    Parametric bore shape: power-law taper, optional bell accent, and optional bubbles.

    Genome layout:
    - [0]: length (min_length .. max_length)
    - [1]: bell diameter (d_bell_min .. d_bell_max)
    - [2]: power exponent for taper curve
    - [3 + 3*k : 3 + 3*(k+1)] for each bubble k: position, width, height
    - [geo_offset + 2*j], [geo_offset + 2*j + 1]: x and y (diameter) offsets for segment j
    """

    def __init__(
        self,
        n_segments: int = 24,
        d0: float = 32,
        d_bell_min: float = 50,
        d_bell_max: float = 80,
        max_length: float = 1900,
        min_length: float = 1500,
        n_bubbles: int = 0,
        smoothness: float = 0.3,
        bell_accent: float = 0.0,
        bell_start: float = 300,
        n_bell_segments: int = 10,
        forced_diameters: Optional[List[Tuple[float, float, float]]] = None,        sine_shape_n : float = 0.0,
        sine_shape_y : float = 0.0
    ):
        """
        Args:
            n_segments: Number of bore segments (excluding endpoints) for the base taper.
            d0: Mouthpiece (input) diameter in mm.
            d_bell_min: Minimum bell diameter in mm (genome maps [0,1] to [d_bell_min, d_bell_max]).
            d_bell_max: Maximum bell diameter in mm.
            max_length: Maximum bore length in mm.
            min_length: Minimum bore length in mm.
            n_bubbles: Number of optional bulge (bubble) features along the bore.
            smoothness: Fraction of bell size used to scale segment y-offsets (0..1).
            bell_accent: If > 0, the last bell_start mm of the bore is accentuated by this factor.
            bell_start: Length (mm) from the bell end over which the bell accent is applied.
            n_bell_segments: Number of points used to build the accentuated bell curve.
            forced_diameters: A list of triples [(pos_pct, diam, length), ...] defining 
                geometric constraints along the bore:
                - pos_pct (float): The longitudinal position expressed as a percentage of 
                the total bore length (0 = mouthpiece, 1 = bell end).
                - diam (float): The target internal diameter in mm to be enforced 
                at that position.
                - length (float): The longitudinal distance (mm) over which this diameter 
                should be maintained. If > 0, the code creates a cylindrical section 
                centered at the calculated position.            
        """

        self.max_length = max_length
        self.min_length = min_length
        self.n_segments = n_segments
        self.d0 = d0
        self.d_bell_min = d_bell_min
        self.d_bell_max = d_bell_max
        self.n_bubbles = n_bubbles
        self.smoothness = smoothness
        self.bell_accent = bell_accent
        self.bell_start = bell_start
        self.n_bell_segments = n_bell_segments
        if forced_diameters is None or len(forced_diameters) == 0:
            self.forced_constraints = []
        else:
            self.forced_constraints = list(forced_diameters)
        self.sine_shape_n : float = sine_shape_n
        self.sine_shape_y : float = sine_shape_y

        self.bubble_width = 300
        self.bubble_height = 40
        self.geo_offset = 3 + self.n_bubbles * 3
        genome_length = 3 + 2 * (n_segments - 1) + self.n_bubbles * 3
        self.n_bubble_segments = 10

        GeoGenome.__init__(self, n_genes=genome_length)


    def get_properties(self) -> Tuple[float, float, float, np.ndarray, np.ndarray, List]:
        """
        Decode genome into length, bell size, power, segment offsets, and bubble list.

        Returns:
            Tuple of (length_mm, bell_size_mm, power, x_genome, y_genome, bubbles).
            bubbles is a list of (pos, width, height) per bubble.
        """
        length = self.genome[0] * (self.max_length - self.min_length) + self.min_length
        bell_size = self.genome[1] * (self.d_bell_max - self.d_bell_min) + self.d_bell_min
        power = self.genome[2] * 4

        bubbles = []
        j = 3
        for i in range(self.n_bubbles):
            pos = self.bubble_width + self.genome[j] * (length - 2 * self.bubble_width)
            width = self.bubble_width * (0.2 + self.genome[j + 1]) / 1.2
            height = (0.2 + self.genome[j + 2]) * self.bubble_height / 1.2
            j += 3
            bubbles.append((pos, width, height))

        # Segment x/y encoded as alternating genome slots from geo_offset
        x_genome = np.array([self.genome[i] for i in range(self.geo_offset, len(self.genome), 2)])
        y_genome = np.array([self.genome[i] for i in range(self.geo_offset + 1, len(self.genome), 2)])

        return length, bell_size, power, x_genome, y_genome, bubbles

    def force_diameters(self, x, y, length):
        if len(self.forced_constraints) > 0:
            # We will use RBF to find the 'deformation' required to hit targets
            target_pts = []
            target_diams = []
            
            for pos_pct, diam, f_length in self.forced_constraints:
                # API sends pos_pct 0-100; KigaliShape expects 0-1
                pct = pos_pct / 100.0 if pos_pct > 1 else pos_pct
                center_x = pct * length
                
                if f_length > 0:
                    # To ensure it's a perfect cylinder, we sample multiple points 
                    # along the forced length to "overpower" any local jitter.
                    half_len = f_length / 2
                    side_pts = np.linspace(center_x - half_len, center_x + half_len, 5)
                    for p in side_pts:
                        target_pts.append(p)
                        target_diams.append(diam)
                else:
                    target_pts.append(center_x)
                    target_diams.append(diam)

            # Create a temporary geometry to see how far off the current "genetic" bore is
            temp_geo = Geo(list(zip(x, y)))
            geo_max_x = max(seg[0] for seg in temp_geo.geo)
            f_x = np.array(target_pts)
            f_d_target = np.array(target_diams)
            # Clamp target x to valid bore range to avoid diameter_at_x assert
            eps = 1e-9 * max(geo_max_x, 1.0)
            upper = geo_max_x - eps if geo_max_x > eps else geo_max_x
            f_x_clamped = np.clip(f_x, 0.0, upper)

            # Calculate the difference (delta) between target and current evolved shape
            f_d_current = np.array([Geo.diameter_at_x(temp_geo, xi) for xi in f_x_clamped])
            deltas = f_d_target - f_d_current
            
            # Anchor mouthpiece and bell (0 delta) so they remain at genetic values
            # Use clamped x so RBF (x, delta) pairs stay within valid bore range
            all_f_x = np.concatenate([[0], f_x_clamped, [geo_max_x]])
            all_deltas = np.concatenate([[0], deltas, [0]])
            
            # RBF Interpolation with 'thin_plate' creates a smooth 'warp' function
            # that hits your deltas exactly at the specified points.
            itp = Rbf(all_f_x, all_deltas, function='thin_plate')
            
            # Apply the warp to the diameter array
            y += itp(x)

        return x,y

    def genome2geo(self) -> Geo:
        length, bell_size, power, x_genome, y_genome, bubbles = self.get_properties()

        # 1. Generate Base Geometry (Power-law taper)
        x = np.linspace(0, length, self.n_segments + 1)
        y = np.power(np.linspace(0, 1, self.n_segments + 1), power)
        # Linear scale from mouthpiece to bell_size
        y = y * (bell_size - self.d0) + self.d0

        # Apply genome jitter/offsets
        shift_x = length / self.n_segments
        x += np.concatenate(([0], (x_genome - 0.5) * shift_x, [0]))
        shift_y = (1 - self.smoothness) * bell_size
        y += np.concatenate(([0], 0.3 * (y_genome - 0.5) * shift_y, [0]))

        # 2. Apply Bell Accent
        x, y = self.apply_bell_accent(x, y, length, bell_size)

        # 3. Add Bubbles
        for bubble in bubbles:
            # Unpack the tuple: (position, width, height)
            pos, width, height = bubble
            x, y = self.make_bubble(x, y, pos, width, height)

        x, y = self.add_sine_shape(x, y)
        
        # 4. Force Exact Diameters (RBF)
        x, y = self.force_diameters(x, y, length)

        # 5. Finalize (Clamp to ensure d_bell_max is respected)
        # We clamp to bell_size specifically here
        x, y = self.fix_didge(x, y, self.d0, bell_size)
        
        return Geo(list(zip(x, y)))

    def fix_didge(
        self, x: np.ndarray, y: np.ndarray, d0: float, bellsize: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clamp bore diameter to [0.9*d0, 1.3*bellsize] to avoid invalid geometry.

        Args:
            x: Array of x positions along the bore (mm).
            y: Array of bore diameters at each x (mm).
            d0: Mouthpiece diameter in mm; minimum allowed is 0.9*d0.
            bellsize: Bell diameter in mm; maximum allowed is 1.3*bellsize.

        Returns:
            Tuple (x, y) of arrays with y clamped; x unchanged.
        """
        mind = d0 * 0.9
        x, y = x.copy(), y.copy()
        y[y < mind] = mind
        y[y > bellsize * 1.3] = bellsize * 1.3
        return x, y

    def make_bubble(
        self,
        x: np.ndarray,
        y: np.ndarray,
        pos: float,
        width: float,
        height: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Insert a sinusoidal bulge (bubble) at pos with given width and height.

        The bubble is clamped so it does not extend past the bore ends. Bore diameter
        inside the bubble is existing diameter plus a half-sine bulge of given height.

        Args:
            x: Array of x positions along the bore (mm).
            y: Array of bore diameters at each x (mm).
            pos: Center x position of the bubble (mm).
            width: Length of the bubble along the bore (mm).
            height: Amplitude of the bulge in mm (added to local bore diameter).

        Returns:
            Tuple (x_new, y_new) of arrays with the segment in [pos-width/2, pos+width/2]
            replaced by bubble points.
        """
        bubble_start_x = pos - 0.5 * width
        bubble_end_x = pos + 0.5 * width

        if bubble_start_x < 20:
            diff = 20-bubble_start_x
            bubble_start_x += diff
            bubble_end_x += diff

        if bubble_end_x > x[-1]:
            diff = bubble_end_x-x[-1]
            bubble_start_x -= diff
            bubble_end_x -= diff

        bubble_x = np.linspace(
            bubble_start_x, bubble_end_x, self.n_bubble_segments, endpoint=False
        )

        # Bore diameter at bubble x, plus half-sine bulge
        geo = Geo(list(zip(x, y)))
        bubble_y_1 = np.array([Geo.diameter_at_x(geo, xi) for xi in bubble_x])
        bubble_y_2 = height * np.sin(
            np.pi * np.arange(self.n_bubble_segments) / self.n_bubble_segments
        )
        bubble_y = bubble_y_1 + bubble_y_2

        # Replace segment inside [bubble_start_x, bubble_end_x] with bubble points
        y = np.concatenate((y[x < bubble_start_x], bubble_y, y[x > bubble_end_x]))
        x = np.concatenate((x[x < bubble_start_x], bubble_x, x[x > bubble_end_x]))
        return x, y

    def apply_bell_accent(self, x: np.ndarray, y: np.ndarray, length: float, bell_size: float) -> np.ndarray:
        """
        Applies a flare to the end of the didgeridoo. 
        If bell_accent is 0, it follows the existing taper.
        If bell_accent is 1, it adds a sharp parabolic flare.
        """
        if self.bell_accent <= 0:
            return x, y

        if self.bell_accent > 1:
            self.bell_accent = 1

        # insert a segment at bell_start position
        bell_start = x[-1] - self.bell_start
        if bell_start < 1:
            bell_start = 1

        if bell_start not in x:
            segments = list(zip(x,y))
            geo = Geo(segments)
            bell_start_diameter = Geo.diameter_at_x(geo, bell_start)
            segments.append([bell_start, bell_start_diameter])
            segments = sorted(segments, key=lambda x:x[0])
            segments = np.array(segments)
            x = segments[:,0]
            y = segments[:,1]

        bell_size_before = y[-1]

        y = (1-0.7*self.bell_accent)*(y-self.d0)
        y += self.d0

        segments = list(zip(x,y))
        geo = Geo(segments)

        x_bell = np.linspace(bell_start, length, self.n_bell_segments)
        y_bell = [Geo.diameter_at_x(geo, _x) for _x in x_bell]

        x_sigmoid = np.linspace(-5, 5., self.n_bell_segments)
        delta = (bell_size_before-y[-1]) * (1-(1/(1 + np.exp(x_sigmoid)))) # sigmoid function
        y_bell += delta

        i1 = x<bell_start

        x = np.concatenate((x[i1], x_bell))
        y = np.concatenate((y[i1], y_bell))

        return x, y

    def add_sine_shape(self, x, y):

        if self.sine_shape_n <= 0 or self.sine_shape_y <= 0:
            return x, y

        x_sine = 2*self.sine_shape_n*np.pi*x/x.max()
        y_sine = np.sin(x_sine)
        y_sine[y_sine<0] = 0

        y += self.sine_shape_y * y_sine
        return x,y
