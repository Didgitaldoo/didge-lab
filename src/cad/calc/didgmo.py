import os
import subprocess
import math
import copy
from prettytable import PrettyTable
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm
from threading import Lock
from cad.calc.geo import Geo

class PeakFile:

    def __init__(self, infile):

        self.impedance_peaks=[]
        self.groundtone_peaks=[]
        self.first_overblow_peaks=[]

        f=open(infile)
        for line in f:
            line=line[0:-1].split(" ")

            note_name=line[2]
            number=int(note_name[-1])
            number-=2
            note_name=note_name[0:-1] + str(number)
            peak={
                "t": int(line[0]),
                "freq": int(line[1]),
                "note": note_name,
                "cent-diff": int(line[3]),
                "amp": float(line[4])
            }

            if peak["t"]==0:
                self.impedance_peaks.append(peak)
            elif peak["t"]==1:
                self.groundtone_peaks.append(peak)
            elif peak["t"]==2:
                self.first_overblow_peaks.append(peak)
            else:
                raise Exception("unknown t")
        f.close()

    def get_drone_freq(self):
        return self.groundtone_peaks[0]["freq"]

    def print_impedance_peaks(self, limit=None):

        s=""
        c=0
        for p in self.impedance_peaks:

            c+=1

            s += "{%s|%.03d|%.02d}, " % (p["note"], p["freq"], p["cent-diff"])

            if limit != None and c==limit:
                break
        print(s[0:-2])

    def get_impedance_table(self, limit=None):

        df={}
        for key in self.impedance_peaks[0].keys():
            df[key]=[]

        for p in self.impedance_peaks:

            for key in p.keys():
                df[key].append(p[key])
        
        df=pd.DataFrame(df)
        a0=df["amp"][0]
        df["amp_relative"]=df["amp"]/a0
        return df

lock=Lock()
def didgmo_bridge(geo : Geo, skip_fft=False):

    lock.acquire()
    file_num=0
    while os.path.exists("temp" + str(file_num) + ".geo"):
        file_num+=1
    lock.release()

    name="temp" + str(file_num)
    outfile=name + ".geo"
    new_geo=geo.copy()
    new_geo.scale(0.001)
    try:
        new_geo.write_geo(outfile)
        command=["didgmo", "geo2fft", name, "1000"]
        subprocess.check_output(command)
        if not skip_fft:
            fft=pd.read_csv(name + ".fft", delimiter=" ", names=["freq", "impedance", "ground", "overblow"])
            peak=PeakFile(name + ".peak")
            return peak, fft
        else:
            peak=PeakFile(name + ".peak")
            return peak
    finally:
        files=[outfile, name + ".fft", name + ".peak", name + ".lab"]
        for f in files:
            os.remove(f)

def cleanup():

    files=os.listdir(".")
    for f in files:
        if f[0:4]=="temp":
            os.remove(f)