# cadsd python implementation

import math
import cmath
import numpy as np
import pandas as pd
import cython

print("using non-cython version of cadsd")

# float64 would be better but it gave an error on MacOS
p = np.float64(1.2929)
n = np.float64(1.708e-5)
c = np.float64(343.37)
PI=np.float64(np.pi)

#c = 331.45 * sqrt(293.16 / 273.16)
class Segment:

    def __init__(self, L, d0, d1):
        self.L = L
        self.d0 = d0
        self.d1 = d1

        self.a0 = PI * d0 * d0 / 4
        self.a01 = PI * (d0 + d1) * (d0 + d1) / 16
        self.a1 = PI * d1 * d1 / 4
        self.phi = math.atan ((d1 - d0) / (2 * L))

        x = (2 * math.sin (self.phi))
        if x==0:
            x=1e-20
        self.l = (d1 - d0) / x
        self.x1 = d1 / x
        self.x0 = self.x1 - self.l
        self.r0 = p * c / self.a0

    @classmethod
    def create_segments_from_geo(cls, geo):

        segments=[]
        shape=[[np.float64(x)[0]/1000, np.float64(x)[1]/1000] for x in geo]
        for i_seg in range(1, len(shape)):
            seg1=shape[i_seg]
            seg0=shape[i_seg-1]
            L=seg1[0]-seg0[0]
            d0=seg0[1]
            d1=seg1[1]
            seg=Segment(L, d0, d1)
            segments.append(seg)
        return segments

def create_segments_from_geo(geo):
    return Segment.create_segments_from_geo(geo)


def ap (w, segments):

    x=[[1,0],[0,1]]
    y=[[0,0], [0,0]]
    z=[[0,0],[0,0]]

    for t_seg in segments:

        L=t_seg.L
        d0=t_seg.d0
        d1=t_seg.d1

        a0 = t_seg.a0
        a01 = t_seg.a01
        a1 = t_seg.a1
        l = t_seg.l
        x0 = t_seg.x0
        x1 = t_seg.x1
        r0 = t_seg.r0

        rvw = math.sqrt (p * w * a01 / (n * PI))
        kw = w / c
        Tw =np.complex128(kw * 1.045 / rvw + (kw * (1.0 + 1.045 / rvw))*1j)
        Zcw = np.complex128(r0 * (1.0 + 0.369 / rvw) - 1j*r0 * 0.369 / rvw)

        ccoshlwl = cmath.cosh(Tw * l)
        csinhlwl = cmath.sinh(Tw * l)
        ccoshlwL = cmath.cosh(Tw * L)
        csinhlwL = cmath.sinh(Tw * L)

        if (d0 != d1):
            y[0][0] = x1 / x0 * (ccoshlwl - csinhlwl / (Tw * x1))
            y[0][1] = x0 / x1 * Zcw * csinhlwl
            y[1][0] = ((x1 / x0 - 1.0 / (Tw * Tw * x0 * x0)) * csinhlwl + Tw * l / ((Tw * x0) * (Tw * x0)) * ccoshlwl) / Zcw
            y[1][1] = x0 / x1 * (ccoshlwl + csinhlwl / (Tw * x0))
        else:
            y[0][0] = ccoshlwL
            y[0][1] = Zcw * csinhlwL
            y[1][0] = csinhlwL / Zcw
            y[1][1] = ccoshlwL

        # dot product
        z[0][0] = x[0][0] * y[0][0] + x[0][1] * y[1][0]
        z[0][1] = x[0][0] * y[0][1] + x[0][1] * y[1][1]
        z[1][0] = x[1][0] * y[0][0] + x[1][1] * y[1][0]
        z[1][1] = x[1][0] * y[0][1] + x[1][1] * y[1][1]

        x[0][0] = z[0][0]
        x[0][1] = z[0][1]
        x[1][0] = z[1][0]
        x[1][1] = z[1][1]

    return z

def Za( w, segments):
    t_seg=segments[-1]

    L = t_seg.L
    d1 = t_seg.d1
    a01 = t_seg.a01
    r0 = t_seg.r0

    rvw = math.sqrt (p * w * a01 / (n * PI))
    Zcw = np.complex128(r0*(1.0 + 0.369 / rvw) - 1j*r0 * 0.369 / rvw)

    res = 0.5 * Zcw * np.complex128(w * w * d1 * d1 / c / c + 1j*0.6 * L * w * d1 / c)	# from geipel

    return res

def cadsd_Ze (segments, f):
  w = 2.0 * PI * f
  a = Za (w, segments)

  b = ap (w, segments)

  Ze = abs ((a * b[0][0] + b[0][1]) / (a * b[1][0] + b[1][1]))
  return Ze


def geo_fft (geo, gmax, offset):

    fft={
        "impedance": {},
        "overblow": {},
        "ground": {}
    }

    for key in fft.keys(): 
        fft[key][0]=0

    segments=Segment.create_segments_from_geo(geo)
    for f in range(1, gmax):
        fft["impedance"][f] = cadsd_Ze(segments, f)
        fft["overblow"][f] = 0
        fft["ground"][f] = 0


    # search for peaks and valleys
    peaks=[0,0]
    vally=[0,0]

    up=False
    npeaks=0
    nvally=0

    freqs=fft["impedance"].keys()
    for i in range(2, len(freqs)):
        if fft["impedance"][i] > fft["impedance"][i-1]:
            if npeaks and not up:
                vally[nvally]=i-1
                nvally+=1
            up=True
        else:
            if up:
                peaks[npeaks] = i-1
                npeaks+=1
            up=False 

        if nvally>1:
            break

    if peaks[0]<0:
        return None

    k = 0.0001

    mem0 = peaks[0]
    mem0a = peaks[0]

    mem0b = mem0a   

    return fft


    # # calculate overblow spectrum of base tone

    # i=mem0
    # while i<gmax:
    #     for j in range( -1*mem0a, mem0b):
    #         if i + j < gmax:
    #             if j < 0:
    #                 fft["ground"][i + j + offset] += fft["impedance"][mem0 + j] * math.exp(i*k)
    #             else: 
    #                 fft["ground"][i + j + offset] += fft["impedance"][mem0-j] * math.exp(i*k)
    #     i+=mem0

    # for i in range(0, gmax):
    #     fft["ground"][i] = fft["impedance"][i] + fft["ground"][i]*1e-6

    # mem1 = peaks[1]

    # mem1a = peaks[1] - vally[0]

    # mem1b = vally[1] - peaks[1]
    # mem1b = mem1a

    # # calculate overblow spectrum of first overblow

    # i = mem1
    # while i<gmax:
    #     for j in range(-mem1a, mem1b):
    #         if i + j < gmax:
    #             if j < 0:
    #                 fft["overblow"][i + j + offset] += fft["impedance"][mem1+j]*mathexp(i*k)
    #             else:
    #                 fft["overblow"][i+j+offset] += fft["impedance"][mem1-j]*math.exp(i*k)
    #     i+=mem1




    # #calculate sound spectrum of first overblow
    # for i in range(gmax):
    #     fft["overblow"][i] = fft["impedance"][i] * fft["overblow"][i] * 1e-6

    # return fft
