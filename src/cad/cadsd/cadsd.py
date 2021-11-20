import pyximport; pyximport.install()
import cad.cadsd._cadsd as cadsd
#import cad.cadsd._cadsd_py as cadsd
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from cad.calc.conv import freq_to_note_and_cent, note_name

class CADSDResult():

    def __init__(self, fft, peaks, geo):
        self.fft=fft
        self.peaks=peaks
        self.geo=geo

    @classmethod
    def from_geo(cls, geo):
        fft=get_highres_spektrum(geo.geo)
        peaks=get_peaks(fft)
        return CADSDResult(fft, peaks, geo)

    def print_summary(self, loss=None):
        s=f"length:\t\t{self.geo.length():.2f}\n"
        s+=f"bell size:\t{self.geo.geo[-1][1]:.2f}\n"
        s+=f"num segments:\t{len(self.geo.geo)}\n"
        s+=f"num peaks:\t{len(self.peaks)}\n"
        if loss != None:
            s+=f"loss:\t\t{loss:.2f}\n"
            
        s+=str(self.peaks)
        print(s)


def get_peaks(fft):
    peaks = fft.iloc[argrelextrema(fft.impedance.values, np.greater_equal)[0]].copy()
    t=[freq_to_note_and_cent(x) for x in peaks["freq"]]
    peaks["note-number"], peaks["cent-diff"]=zip(*t)
    peaks["note-name"] = peaks["note-number"].apply(lambda x : note_name(x))
    return peaks

def get_impedance_spektrum(geo, from_freq, to_freq, stepsize):

    segments=cadsd.Segment.create_segments_from_geo(geo)
    spektrum={
        "freq": [],
        "impedance": []
    }

    for freq in np.arange(from_freq, to_freq, stepsize):
        spektrum["freq"].append(freq)
        impedance=cadsd.cadsd_Ze(segments, freq)
        spektrum["impedance"].append(impedance)
    
    return pd.DataFrame(spektrum)

def get_highres_spektrum(geo):

    df1=get_impedance_spektrum(geo, 30, 100, 0.1)
    df2=get_impedance_spektrum(geo, 101, 1000, 1)
    return pd.concat([df1, df2])