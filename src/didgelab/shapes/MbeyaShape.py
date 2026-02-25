"""
Mbeya-style parametric didgeridoo shape (MbeyaGenome).

Genome with named parameters: straight section, opening section, bell, and
optional bubbles. Each parameter has min/max and is encoded in the genome.
"""

import numpy as np

from ..evo import GeoGenome
from ..geo import Geo


class MbeyaGenome(GeoGenome):
    """
    Parametric bore: straight tube, opening section, bell, then optional bubbles.

    Uses named parameters (add_param / get_value) so each gene has a min/max.
    Genome length = number of named params. build: mouth -> straight -> opening -> bell -> bubbles.
    """

    def add_param(self, name: str, minval: float, maxval: float) -> None:
        """
        Register a named parameter; next genome index, linear map [0,1] -> [minval, maxval].

        Args:
            name: Parameter name (used with get_value).
            minval: Minimum value when genome gene is 0.
            maxval: Maximum value when genome gene is 1.
        """
        self.named_params[name] = {
            "index": len(self.named_params),
            "min": minval,
            "max": maxval,
        }

    def get_value(self, name: str) -> float:
        """
        Decode genome at this parameter's index to value in [min, max].

        Args:
            name: Parameter name previously registered with add_param.

        Returns:
            Value in the range [minval, maxval] for that parameter.
        """
        p = self.named_params[name]
        v = self.genome[p["index"]]
        return v * (p["max"] - p["min"]) + p["min"]

    def __init__(self, n_bubbles: int = 3, add_bubble_prob: float = 0.7):
        """
        Args:
            n_bubbles: Number of bubble (bulge) features along the bore; each has pos, width, height params.
            add_bubble_prob: Probability used when sampling/evolving whether to add a bubble (0..1).
        """
        self.named_params = {}

        self.d1 = 32
        self.add_bubble_prob = add_bubble_prob
        self.n_bubbles = n_bubbles

        # Straight cylindrical part (length, diameter factor)
        self.add_param("l_gerade", 500, 1500)
        self.add_param("d_gerade", 0.9, 1.2)

        # Opening section (number of segments, x/y power factors, length)
        self.add_param("n_opening_segments", 0, 8)
        self.add_param("opening_factor_x", -2, 2)
        self.add_param("opening_factor_y", -2, 2)
        self.add_param("opening_length", 700, 1000)

        # Bell (pre-bell diameter, bell length, bellsize increment)
        self.add_param("d_pre_bell", 40, 50)
        self.add_param("l_bell", 20, 50)
        self.add_param("bellsize", 5, 30)

        # Per-bubble: height, position (0..1 along length), width
        for i in range(self.n_bubbles):
            self.add_param(f"add_bubble_{i}", 0, 1)
            self.add_param(f"bubble_height_{i}", -0.5, 1)
            self.add_param(f"bubble_pos_{i}", 0, 1)
            self.add_param(f"bubble_width_{i}", 150, 300)

        self.n_segments = 11

        GeoGenome.__init__(self, n_genes=len(self.named_params))

    def make_bubble(
        self, shape: list, pos: float, width: float, height: float
    ) -> list:
        """
        Insert a sinusoidal bulge into shape: points before bubble, bubble arc, points after.

        Args:
            shape: List of [x, diameter] segment points (mm) defining the current bore.
            pos: Center x position of the bubble (mm).
            width: Length of the bubble along the bore (mm).
            height: Bulge factor; local diameter is multiplied by 1 + sin(pi*j/n)*height over the bubble.

        Returns:
            New list of [x, diameter] points with the bubble segment replaced by the bulge arc.
        """
        i = self.get_index(shape, pos - 0.5 * width)
        bubbleshape = shape[0:i]

        x = pos - 0.5 * width
        y = Geo.diameter_at_x(shape, x)
        if shape[i - 1][0] < x:
            bubbleshape.append([x, y])

        for j in range(1, self.n_segments):
            x = pos - 0.5 * width + j * width / self.n_segments
            y = Geo.diameter_at_x(shape, x)
            factor = 1 + np.sin(j * np.pi / self.n_segments) * height
            y *= factor
            bubbleshape.append([x, y])

        x = pos + 0.5 * width
        y = Geo.diameter_at_x(shape, x)
        bubbleshape.append([x, y])

        while i < len(shape) and shape[i][0] <= bubbleshape[-1][0] + 1:
            i += 1
        bubbleshape.extend(shape[i:])

        return bubbleshape

    def get_index(self, shape: list, x: float) -> int:
        """
        Return the first index i such that shape[i][0] > x (or len(shape)-1).

        Args:
            shape: List of [x, diameter] segment points (mm), x strictly increasing.
            x: Position in mm along the bore.

        Returns:
            Smallest i with shape[i][0] > x, or len(shape)-1 if no such i.
        """
        for i in range(len(shape)):
            if shape[i][0] > x:
                return i
        return len(shape) - 1

    def genome2geo(self) -> Geo:
        """
        Build geometry: mouth -> straight -> opening (power-law) -> bell -> bubbles.

        Returns:
            Geo instance with bore as list of (x_mm, diameter_mm) segments; zero-length
            segments removed via Geo.fix_zero_length_segments.
        """
        shape = [[0, self.d1]]

        # Straight section
        p = [
            self.get_value("l_gerade"),
            shape[-1][1] * self.get_value("d_gerade"),
        ]
        shape.append(p)

        # Opening section: power-law spacing in x and y, then normalised and scaled
        n_seg = int(self.get_value("n_opening_segments"))
        seg_x = []
        seg_y = []
        for i in range(n_seg):
            seg_x.append(pow(i + 1, self.get_value("opening_factor_x")))
            seg_y.append(pow(i + 1, self.get_value("opening_factor_y")))

        def normalize(arr):
            m = sum(arr)
            return [x / m for x in arr]

        seg_x = normalize(seg_x)
        seg_y = normalize(seg_y)
        seg_x = [x * self.get_value("opening_length") for x in seg_x]
        seg_y = [y * self.get_value("d_pre_bell") for y in seg_y]

        start_x, start_y = shape[-1][0], shape[-1][1]
        for i in range(n_seg):
            shape.append([
                sum(seg_x[0 : i + 1]) + start_x,
                sum(seg_y[0 : i + 1]) + start_y,
            ])

        # Bell
        shape.append([
            shape[-1][0] + self.get_value("l_bell"),
            shape[-1][1] + self.get_value("bellsize"),
        ])

        # Bubbles (position clamped so bubble stays inside bore)
        for i in range(self.n_bubbles):
            pos = shape[-1][0] * self.get_value(f"bubble_pos_{i}")
            width = self.get_value(f"bubble_width_{i}")
            height = self.get_value(f"bubble_height_{i}")
            if pos - width / 2 < -10:
                pos = width / 2 + 10
            if pos + width / 2 + 10 > shape[-1][0]:
                pos = shape[-1][0] - width / 2 - 10
            shape = self.make_bubble(shape, pos, width, height)

        geo = Geo(shape)
        geo = Geo.fix_zero_length_segments(geo)
        return geo

