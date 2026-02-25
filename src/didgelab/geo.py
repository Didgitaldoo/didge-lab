"""
Didgeridoo geometry representation and operations for DidgeLab.

A geometry is a list of segments, each (x, diameter) in mm: distance from
mouthpiece and bore diameter. Supports loading/saving, scaling, cones,
bubbles, volume, and interpolation of diameter along the bore.
"""

import copy
import pandas as pd
import math
import json


class Geo:
    """
    Bore geometry of a didgeridoo as a list of (x, diameter) segments in mm.

    Attributes:
        geo: List of [x, d] with x = distance from mouthpiece, d = diameter.
    """

    def __init__(self, geo=None, infile=None):
        self.geo=[]

        if infile != None:
            self.geo = self.read_geo(infile)

        if geo != None:
            self.geo=geo

        # remove zero length segments
        clean_geo=[]
        for i in range(0, len(self.geo)):
            if i>0 and self.geo[i][0]==self.geo[i-1][0]:
                continue
            clean_geo.append(self.geo[i])
        self.geo = clean_geo

    @classmethod
    def make_cone(cls, length, d1, d2, n_segments):
        """Build a conical bore: length (mm), diameters d1 (mouth), d2 (bell), n_segments."""
        shape=[]
        shape.append([0, d1])

        z=(d2-d1)/2
        angle=math.atan(z/length)
        for i in range(1, n_segments):
            x=length*i/(n_segments-1)
            y=2*x*math.tan(angle) + d1
            shape.append([x,y])
        return Geo(geo=shape)

    def read_geo(self, infile):
        """Load geometry from a JSON file (list of [x, d] pairs)."""
        f=open(infile)
        geo = json.load(f)
        f.close()
        return geo

    def write_geo(self, outfile):
        """Write geometry to a text file (one 'x d' line per segment)."""
        f=open(outfile, "w")
        for segment in self.geo:
            seg=f"{segment[0]:.10f} {segment[1]:.10f}\n"
            f.write(seg)
        f.close()

    def stretch(self, factor):
        """Scale all x (length) by factor; diameters unchanged."""
        for i in range(0, len(self.geo)):
            self.geo[i][0]*=factor

    def copy(self):
        geo=copy.deepcopy(self.geo)
        return Geo(geo=geo)

    def scale(self, factor):
        """Scale all x and diameter by factor (e.g. mm to m)."""
        for i in range(0, len(self.geo)):
            self.geo[i][0]*=factor
            self.geo[i][1] *= factor

    def make_bubble(self, pos, width, height):
        """Insert a bulge (bubble) at position pos with given width and height."""
        index=0
        for i in range(len(self.geo)):
            if self.geo[i+1][0]>pos:
                index=i
                break
        
        left=self.geo[0:index+1]
        right=self.geo[index+1:]
        new_geo=left + [
            [pos-width/2, self.geo[index][1]],
            [pos, height],
            [pos+width/2, self.geo[index+1][1]],
        ] + right
        self.geo = new_geo

    def move_segments_x(self, start, end, offset):
        """Shift x-coordinates of segments [start..end] by offset."""
        for i in range(start, end+1):
            self.geo[i][0]+=offset
            pass

    def length(self):
        """Total length in mm (x of last segment)."""
        return self.geo[-1][0]

    def bellsize(self):
        """Bell diameter in mm (diameter of last segment)."""
        return self.geo[-1][1]

    def segments_to_str(self):
        df={}
        for x in ["x", "y"]:
            df[x]=[]
        for i in range(0, len(self.geo)):
            df["x"].append(int(self.geo[i][0]))
            df["y"].append(int(self.geo[i][1]))
        df=pd.DataFrame(df)
        return str(df)

    def sort_segments(self):
        """Sort segments by x (distance from mouthpiece)."""
        self.geo = sorted(self.geo, key=lambda x: x[0])

    def diameter_at_x(self, x):
        """Interpolated bore diameter at position x (mm)."""
        return Geo.diameter_at_x(self, x)

    def compute_volume(self):
        """Approximate internal volume in mm³ (trapezoidal rule along bore)."""
        v = 0
        for i in range(1, len(self.geo)):
            l = self.geo[i][0] - self.geo[i-1][0]
            v+= l*self.geo[i-1][1]
            v += l*(self.geo[i][1] - self.geo[i-1][1])
        """Approximate internal volume (mm³) using trapezoidal rule."""
        return v

    @staticmethod
    def scale_length(geo, max_length):
        """Scale geometry length to max_length (factor applied to x only)."""
        factor = max_length / geo.length()
        for i in range(len(geo.geo)):
            geo.geo[i][0]*=factor
        return geo

    @staticmethod
    def get_max_d(geo):
        """Maximum bore diameter in geometry (mm)."""
        return max([x[1] for x in geo.geo])

    @staticmethod
    def scale_diameter(geo, max_d):
        """Scale diameters so maximum diameter becomes max_d."""
        didge_d = Geo.get_max_d(geo)
        factor=max_d/didge_d
        for i in range(1,len(geo.geo)):
            geo.geo[i][1]*=factor
        return geo

    @staticmethod
    def print_geo_summary(geo, peak=None, loss=None):
        """Format a short text summary of the geometry (and optional peak/loss)."""
        s = f"length:\t\t{geo.length():.2f}\n"
        s+=f"bell size:\t{geo.geo[-1][1]:.2f}\n"
        s+=f"num segments:\t{len(geo.geo)}\n"

        if peak is not None:
            s+=f"num peaks:\t{len(peak)}\n"
        if loss != None:
            s+=f"loss:\t\t{loss:.2f}\n"
            
        s+=str(peak)
        return s

    @staticmethod
    def geo_to_json(geo):
        return geo.geo

    @staticmethod
    def json_to_geo(geo_json):
        return Geo(geo=geo_json)

    @staticmethod
    def diameter_at_x(geo, x):
        """Return interpolated bore diameter at position x (mm). geo may be Geo or list of segments."""
        if type(geo) == Geo:
            geo=geo.geo

        assert x<=geo[-1][0]

        if x==0:
            return geo[0][1]

        for i in range(len(geo)):
            if x<geo[i][0]:
                break

        x1=geo[i-1][0]
        y1=geo[i-1][1]
        x2=geo[i][0]
        y2=geo[i][1]

        ydiff=(y2-y1)/2
        xdiff=(x2-x1)

        winkel=math.atan(ydiff/xdiff)
        y=math.tan(winkel)*(x-x1)*2+y1
        return y

    @staticmethod
    def fix_zero_length_segments(geo):
        """Remove segments that have zero length (same x as previous)."""
        fix = None
        new_geo=[geo.geo[0]]
        for i in range(1, len(geo.geo)):
            if geo.geo[i][0]-new_geo[-1][0] > 0:
                new_geo.append(geo.geo[i])
        return Geo(new_geo)

