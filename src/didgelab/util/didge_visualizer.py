import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import os
import json
from didgelab.calc.sim.sim import quick_analysis

# paint a picture of a didgeridoo
class DidgeVisualizer:
    
    @classmethod
    def create_didge_shape(cls, geo):

        margin_y=10
        margin_x=10

        max_y=max([x[1] for x in geo.geo])
        center_y = max_y/2 + margin_y

        df={"x":[], "y": [], "series": []}
        for i in range(0, len(geo.geo)):
            p=geo.geo[i]

            p_oben=center_y + p[1]/2 
            p_unten=center_y - p[1]/2

            df["x"].append(p[0])
            df["y"].append(p_oben)
            df["series"].append("oben")
            df["x"].append(p[0])
            df["y"].append(p_unten)
            df["series"].append("unten")
            df["x"].append(p[0])
            df["y"].append(p_unten)
            df["series"].append("seg" + str(i))
            df["x"].append(p[0])
            df["y"].append(p_oben)
            df["series"].append("seg" + str(i))

        return pd.DataFrame(df)
    
    @classmethod
    def vis_didge(cls, geo):

        if type(geo) == list:
            plt.figure(figsize=(15,2*len(geo)))
            for g in geo:
                for i in range(len(geo)):
                    plt.subplot(len(geo), 2, 2*i+1)
                    DidgeVisualizer.vis_didge(geo[i])
                    plt.title("plot " + str(i))
        else:
            df=DidgeVisualizer.create_didge_shape(geo)
            n_series=len(df["series"].unique())
            palette = ["#000000"]*n_series
            sns.set(rc={'figure.figsize':(15,3)})
            g=sns.lineplot(data=df, x="x", y="y", hue="series", palette=palette)
            g.set(ylim=(0, df["y"].max()))
            g.set(xlim=(0, df["x"].max()))
            g.get_legend().remove()
            g.set_yticks([])
            g.xaxis.set_ticks_position("top")
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            g.set_xlabel("")
            g.set_ylabel("")
            plt.axis('equal')
            return g

def vis_didge(geo):
    DidgeVisualizer.vis_didge(geo)

# # paint an image of fft
# class FFTVisualiser:
    
#     @classmethod
#     def vis_fft_and_target(cls, fft):
        
#         #fft=fft.copy().drop(columns=["ground", "overblow"])
#         fft=fft.copy()
#         fft.reset_index(drop=True, inplace=True) 
#         # for column in fft.columns:
#         #     fft[column]=fft[column] / fft[column].max()

#         sns.set(rc={'figure.figsize':(15,5)})
#         sns.lineplot(data=fft, x="freq", y="impedance")
        

# def visualize_geo_to_files(geo, output_dir, filename, skip_cadsd=False, cadsd_result=None):

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     plt.clf()
#     DidgeVisualizer.vis_didge(geo)
#     geofile=os.path.join(output_dir, filename + "geo.png")
#     plt.savefig(geofile, dpi=500)
#     plt.clf()

#     if not skip_cadsd:

#         if cadsd_result==None:
#             cadsd_result=CADSDResult.from_geo(geo)

#         sns.set(rc={'figure.figsize':(15,5)})
#         sns.lineplot(data=cadsd_result.fft, x="freq", y="impedance")

#         # FFTVisualiser.vis_fft_and_target(cadsd_result.fft)
#         fftfile=os.path.join(output_dir, filename + "fft.png")
#         plt.savefig(fftfile, dpi=500)

#         report=geotools.print_geo_summary(geo, peak=cadsd_result.peaks)

#         f=open(os.path.join(output_dir, filename + "report.txt"), "w")
#         f.write(report)
#         f.write("\n\n")
#         f.write(json.dumps(geo.geo))
#         f.close()

