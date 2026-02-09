"""
Kigali-style parametric didgeridoo shape.

Genome encodes length, bell size, power curve, segment positions/heights,
optional bubbles, and optional bell accent. Used with evolutionary search (e.g. Nuevolution).
"""

import numpy as np
from typing import List, Tuple

from ..geo import Geo
from ..evo.genome import GeoGenome


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
        bell_start: float = 200,
        n_bell_segments: int = 10,
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

        self.bubble_width = 300
        self.bubble_height = 40

        # Genome: 3 global params + 3 per bubble + 2*(n_segments-1) for segment x,y
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

    def genome2geo(self) -> Geo:
        """
        Build Geo from genome: power-law taper, segment offsets, optional bell accent, then bubbles.

        Returns:
            Geo instance with bore as list of (x_mm, diameter_mm) segments.
        """
        length, bell_size, power, x_genome, y_genome, bubbles = self.get_properties()

        # Base taper: x evenly along length, y = power-law from d0 to effective bell size
        # When bell_accent > 0, base taper stops short of full bell so the accent can expand it
        x = np.arange(0, 1, 1 / self.n_segments)
        x = np.concatenate((x, [1]))
        x *= length

        y = np.arange(0, 1, 1 / self.n_segments)
        y = np.concatenate((y, [1]))
        y = np.power(y, power)
        y = y * (bell_size / (1 + self.bell_accent) - self.d0) + self.d0

        # Apply genome-driven offsets (centered at 0.5)
        shift_x = length / self.n_segments
        x += np.concatenate(([0], (x_genome - 0.5) * shift_x, [0]))
        shift_y = (1 - self.smoothness) * bell_size
        y += np.concatenate(([0], 0.3 * (y_genome - 0.5) * shift_y, [0]))

        # Optional: replace the bell end with an accentuated bell curve
        if self.bell_accent > 0 and self.bell_start > 0 and self.n_bell_segments > 0:
            geo = Geo(list(zip(x, y)))
            bell_start_index = 0
            while bell_start_index < len(x) and x[bell_start_index] < length - self.bell_start:
                bell_start_index += 1

            x_bell = np.arange(self.n_bell_segments, dtype=float)
            x_bell = x_bell[1:]
            x_bell /= x_bell.max()
            x_bell *= self.bell_start
            x_bell += length - self.bell_start

            y_bell = np.array([Geo.diameter_at_x(geo, xi) for xi in x_bell])
            mult = np.arange(len(y_bell), dtype=float)
            mult /= mult[-1]
            mult = np.power(mult, 2)
            mult /= mult[-1]
            mult = 1 + self.bell_accent * np.power(mult, 2)
            y_bell *= mult

            x = np.concatenate((x[0:bell_start_index], x_bell))
            y = np.concatenate((y[0:bell_start_index], y_bell))

        x, y = self.fix_didge(x, y, self.d0, bell_size)

        for bubble in bubbles:
            pos, width, height = bubble
            x, y = self.make_bubble(x, y, pos, width, height)

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