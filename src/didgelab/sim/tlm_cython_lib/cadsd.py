# python setup.py build_ext --inplace

#import pyximport; pyximport.install()

import os

if os.getenv('CADSD_BACKEND') == "python":
    import didgelab.sim.tlm_cython_lib.cadsd_py as cadsd_imp
else:
    import didgelab.sim.tlm_cython_lib._cadsd as cadsd_imp
import numpy as np
import pandas as pd

from didgelab.conv import freq_to_note_and_cent, note_name, note_to_freq
from didgelab.app import get_config, get_app
try:
    from .correction_model.correction_model import FrequencyCorrectionModel
except ImportError:
    FrequencyCorrectionModel = None

from abc import abstractmethod

class CADSD():

    def __init__(self, geo):
        self.geo=geo
        self.segments=None
        self.correction_model = None

    def get_segments(self):
        if self.segments==None:
            self.segments=cadsd_imp.create_segments_from_geo(self.geo.geo)
        return self.segments

    def compute_raw_impedance(self, frequencies=None):

        segments=self.get_segments()

        if frequencies is None:
            frequencies = CADSD.get_simulation_frequencies()
            
        impedance = [cadsd_imp.cadsd_Ze(segments, freq) for freq in frequencies]
        impedance = np.array(impedance)
        return frequencies, impedance

    def apply_frequency_correction(self, frequencies):
        correction = get_config()["sim.correction"]
        if correction == "none" or FrequencyCorrectionModel is None:
            return frequencies
        correction_service = get_app().get_service(type(FrequencyCorrectionModel))
        return correction_service.correct(frequencies)

    @abstractmethod    
    def get_simulation_frequencies(
        fmin : float=None,
        fmax : float=None,
        grid_size : float=None,
        grid : float=None
        ):
        if fmin is None:
            fmin = get_config()["sim.fmin"]
        if fmax is None:
            fmax = get_config()["sim.fmax"]
        if grid_size is None:
            grid_size = get_config()["sim.grid_size"]
        if grid is None:
            grid = get_config()["sim.grid"]

        if grid == "even":
            frequencies = np.arange(fmin, fmax, grid_size)
            return frequencies
        elif grid == "log":
            frequencies = []
            stepsize = grid_size/1200
            start_freq = fmin
            end_freq = start_freq
            octave = 0

            while end_freq < fmax:
                notes = np.arange(0,1,stepsize) + octave
                frequencies.extend(start_freq*np.power(2, notes))
                end_freq = frequencies[-1]
                octave += 1
                
            frequencies = np.array(list(filter(lambda x:x<=fmax, frequencies)))
            return frequencies
        else:
            raise Exception()
    

    # helper function for compute_ground
    def _get_closest_index(self, freqs, f):
        for i in range(len(freqs)):
            m2=np.abs(freqs[i]-f)
            if i==0:
                m1=m2
                continue
            if m2>m1:
                return i-1
            m1=m2

        if f>freqs[-1]:
            return len(freqs)
        else:
            return len(freqs)-1
        
    # helper function for compute_ground
    def _find_first_maximum_index(self, impedance):

        peaks=[0,0]
        vally=[0,0]

        up = 0
        npeaks = 0
        nvally = 0

        for i in range(get_config()["sim.fmin"]+1, get_config()["sim.fmax"]):
            if impedance[i] > impedance[i-1]:
                if npeaks and not up:
                    vally[nvally] = i - 1
                    nvally+=1
                up = 1
            else:
                if up:
                    peaks[npeaks] = i - 1
                    npeaks+=1
                up = 0
            if nvally > 1:
                break

        if peaks[0]<0:
            raise Exception("bad fft")

        return peaks[0]
    
    def interpolate_ground(self, freqs, impedance):
        fundamental_i = self._find_first_maximum_index(impedance)
        fundamental_freq = freqs[fundamental_i]

        mem0 = fundamental_freq
        mem0a = fundamental_freq
        mem0b = fundamental_freq
        
        for i in range(mem0, get_config()["sim.fmax"], mem0):
            for j in range(-mem0a, mem0b):
                if i + j < get_config()["sim.fmax"] and i + j + offset>get_config()["sim.fmin"] and mem0-j>=get_config()["sim.fmin"] and mem0+j>=get_config()["sim.fmin"]: 
                    if j < 0:
                        fft["ground"][i + j + offset] += fft["impedance"][mem0 + j] * np.exp (i * k)
                    else:
                        fft["ground"][i + j + offset] += fft["impedance"][mem0 - j] * np.exp (i * k)

        # calculate sound specturm of base tone
        for i in range(get_config()["sim.fmin"], get_config()["sim.fmax"]):
            fft["ground"][i] = fft["impedance"][i] * fft["ground"][i] * 1e-6


        

    # compute ground spektrum from impedance spektrum
    # warning: frequencies must be evenly spaced
    def compute_ground_spektrum(self, freqs, impedance):

        fmin = np.min(freqs)
        fmax = np.max(freqs)

        fundamental_i = self._find_first_maximum_index(impedance)
        fundamental_freq = freqs[fundamental_i]

        print(fmin, fmax, fundamental_freq)

        ground = np.zeros(len(freqs))
        indizes = np.concatenate((np.arange(fmin,fundamental_freq), np.arange(fundamental_freq,fmin-1,-1)))

        print(indizes)

        window_right = impedance[indizes]

        k = 0.0001
        for i in range(fundamental_freq, fmax, fundamental_freq):

            il = self._get_closest_index(freqs, i-fundamental_freq+1)
            ir = np.min((len(freqs)-1, il+len(window_right)))

            window_left = impedance[il:ir]
            if ir-il!=len(window_right):
                window_right = window_right[0:ir-il]

            ground[il:ir] += window_right*np.exp(i*k)

        for i in range(len(ground)):
            ground[i] = impedance[i] * ground[i] * 1e-6

        for i in range(len(ground)):
            x=ground[i]*2e-5
            ground[i] = 0 if x<1 else 20*np.log10(x) 

        return np.array(ground)
    
    def compute_impedance_and_ground(self):
        freq, impedance = self.compute_raw_impedance()
        ground = self.compute_ground_spektrum(freq, impedance)
        impedance *= 1e-6
        return freq, impedance, ground

    def compute_impedance(self):
        freq, impedance = self.compute_raw_impedance()
        impedance *= 1e-6
        return freq, impedance

    def _compute_ground(self):
        freq, impedance = self.compute_raw_impedance()
        

    def get_notes(self):        

        if self.notes is not None:
            return self.notes

        fft=self.get_impedance_spektrum()
        maxima = get_max(fft.freq, fft.impedance, "max")
        peaks=fft.iloc[maxima].copy()
        peaks["rel_imp"]=peaks.impedance / peaks.iloc[0]["impedance"]
        t=[freq_to_note_and_cent(x) for x in peaks["freq"]]
        peaks["note-number"], peaks["cent-diff"]=zip(*t)
        peaks["note-name"] = peaks["note-number"].apply(lambda x : note_name(x))
        self.notes=peaks
        return peaks

    # def get_ground_spektrum(self):
    #     if self.sound_spektra==None:
    #         self._get_sound_spektrum()

    #     return self.sound_spektra["ground"]

    # def get_overblow_spektrum(self):
    #     if self.sound_spektra==None:
    #         self._get_sound_spektrum()

    #     return self.sound_spektra["overblow"]

    # this function could use some optimization
    # 1) split it in ground and overblow spektrum
    # 2) reuse peaks / valley analysis from get_overblow_notes
    # 3) cython

    # parameter offset: frequency offset of ground tone and first overblow
    def _get_sound_spektrum_old(self, offset=0):

        freq, impedance=self.get_impedance_spektrum()

        fft={
            "impedance": dict(zip(freq, impedance)),
            "ground": {},
            "overblow": {}
        }

        fft["impedance"][get_config()["sim.fmin"]]=0
        for i in range(get_config()["sim.fmin"], get_config()["sim.fmax"]):
            fft["ground"][i]=0
            fft["overblow"][i]=0
        
        peaks=[0,0]
        vally=[0,0]

        up = 0
        npeaks = 0
        nvally = 0

        #print(fft["impedance"].keys())
        for i in range(get_config()["sim.fmin"]+1, get_config()["sim.fmax"]):
            if fft["impedance"][i] > fft["impedance"][i-1]:
                if npeaks and not up:
                    vally[nvally] = i - 1
                    nvally+=1
                up = 1
            else:
                if up:
                    peaks[npeaks] = i - 1
                    npeaks+=1
                up = 0
            if nvally > 1:
                break

        if peaks[0]<0:
            raise Exception("bad fft")
        
        k = 0.0001

        mem0 = peaks[0]

        mem0a = peaks[0]

        mem0b = mem0a

        # calculate overblow spectrum of base tone
        for i in range(mem0, get_config()["sim.fmax"], mem0):
            for j in range(-mem0a, mem0b):
                if i + j < get_config()["sim.fmax"] and i + j + offset>get_config()["sim.fmin"] and mem0-j>=get_config()["sim.fmin"] and mem0+j>=get_config()["sim.fmin"]: 
                    if j < 0:
                        fft["ground"][i + j + offset] += fft["impedance"][mem0 + j] * np.exp (i * k)
                    else:
                        fft["ground"][i + j + offset] += fft["impedance"][mem0 - j] * np.exp (i * k)

        # calculate sound specturm of base tone
        for i in range(get_config()["sim.fmin"], get_config()["sim.fmax"]):
            fft["ground"][i] = fft["impedance"][i] * fft["ground"][i] * 1e-6

        mem1 = peaks[1]
        mem1a = peaks[1] - vally[0]
        mem1b = mem1a

        # calculate overblow spectrum of first overblow
        for i in range(mem1, get_config()["sim.fmax"], mem1):
            for j in range(-mem1a, mem1b):
                if i + j < get_config()["sim.fmax"]:
                    if j < 0:
                        fft["overblow"][i + j + offset] += fft["impedance"][mem1 + j] * np.exp (i * k)
                    else:
                        fft["overblow"][i + j + offset] +=fft["impedance"][mem1 - j] * np.exp (i * k)

        # calculate sound spectrum of first overblow
        for i in range(get_config()["sim.fmin"], get_config()["sim.fmax"]):
            fft["overblow"][i] = fft["impedance"][i] * fft["overblow"][i] * 1e-6

        # df={
        #     "freq": fft["ground"].keys(),
        #     "impedance": fft["impedance"].values(),
        #     "ground": fft["ground"].values(),
        #     "overblow": fft["overblow"].values()
        # }
        # df=pd.DataFrame(df)

        # df.impedance=df.impedance.apply(lambda x : x*1e-6)
        # df.ground=df.ground.apply(lambda x : max(0, 20*np.log10(x*2e-5)))
        # df.overblow=df.overblow.apply(lambda x : max(0, 20*np.log10(x*2e-5)))

        for i in range(get_config()["sim.fmin"], get_config()["sim.fmax"]):
            fft["impedance"][i] *= 1e-6
            x=fft["ground"][i]*2e-5
            fft["ground"][i] = 0 if x<1 else 20*np.log10(x) 
            x=fft["overblow"][i]*2e-5
            fft["overblow"][i] = 0 if x<1 else 20*np.log10(x) 


        self.sound_spektra=fft

    # def get_all_spektra_df(self):
    #     if self.all_spektra_df is not None:
    #         return self.all_spektra_df
            
    #     if self.sound_spektra==None:
    #         self._get_sound_spektrum()

    #     self.all_spektra_df={
    #         "freq": self.sound_spektra["ground"].keys(),
    #         "impedance": self.sound_spektra["impedance"].values(),
    #         "ground": self.sound_spektra["ground"].values(),
    #         "overblow": self.sound_spektra["overblow"].values()
    #     }

    #     self.all_spektra_df=pd.DataFrame(self.all_spektra_df)
    #     return self.all_spektra_df

    # def set_additional_metric(self, key, value):
    #     self.additional_metrics[key]=value

    # def get_additional_metric(self, key):
    #     if key not in self.additional_metrics:
    #         return key
    #     else:
    #         return self.additional_metrics[key]

    # def has_additional_metric(self, key):
    #     return key in self.additional_metrics


# find maxima or minima of a numpy array
# scipy.signal import argrelextrema caused problems
# code from https://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
def get_max(x, y, find):
    assert find in ("max", "min")
    a = np.diff(np.sign(np.diff(y))).nonzero()[0] + 1 # local min+max

    if find == "min":
        return (np.diff(np.sign(np.diff(y))) > 0).nonzero()[0] + 1 # local min
    elif find == "max":
        return (np.diff(np.sign(np.diff(y))) < 0).nonzero()[0] + 1 # local max
