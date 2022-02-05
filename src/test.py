from cad.common.app import App
import numpy as np
from cad.calc.geo import geotools
from cad.calc.conv import note_to_freq, note_name, freq_to_note_and_cent
import math
import numpy as np
from cad.calc.geo import Geo
import matplotlib.pyplot as plt
from cad.calc.parameters import FinetuningParameters, MutationParameterSet
import random
from cad.cadsd.cadsd import CADSD, cadsd_volume
from cad.cadsd._cadsd import cadsd_Ze, create_segments_from_geo
from cad.ui.visualization import make_didge_report
from cad.calc.loss import PentaLossFunction
import seaborn as sns
import pandas as pd

class MbeyaShape(MutationParameterSet):

    def __init__(self):
        MutationParameterSet.__init__(self)

        self.d1=32
        self.add_param("l_gerade", 500, 1500)
        self.add_param(("d_gerade", 0.9, 1.5))
		
		self.add_param(("d_gerade", 0.9, 1.5))





#geo=[[0, 32], [119.9958769849766, 33.60786924930998], [182.96338939932383, 37.43993858124537], [229.31303306092116, 37.65868840095003], [243.87150359250768, 44.24922301095391], [248.22661301651374, 44.66872200645895], [273.35700590490666, 45.427288404108324], [325.1833660559223, 45.10078689004896], [326.6977955401382, 38.56326921940415], [331.3271419201785, 39.24648289540542], [348.80183650206857, 43.44272291801225], [430.9911964176373, 36.58658612297562], [467.1100014503583, 38.75489455260514], [656.2303357738663, 43.14580720660788], [674.7045855405672, 42.09763460759996], [788.1166743193038, 45.731774426101126], [800.7700549159376, 48.91416216097986], [874.3223026858111, 57.48849785851295], [894.4633095356007, 57.85511636112676], [920.7665646046486, 62.810078863961245], [924.8861710152711, 60.406541693482154], [965.9223894013795, 52.92986907805038], [993.5297817069403, 47.8899396101002], [1019.7355557222457, 45.38388648475607], [1053.7014100144995, 51.906767868800614], [1056.08323493601, 51.31169232421349], [1167.4257484469804, 61.68647957633659], [1246.3874724224754, 65.64353932959351], [1319.8878127066603, 83.97728075004777], [1349.4583898219896, 115.89351882544594], [1380.2817590427794, 114.68014975992207], [1413.7176032103646, 94.53429022929657], [1469.3136630077106, 124.90807502454308], [1509.405056810938, 137.6973360550101], [1522.1553223428134, 131.80498827396133], [1577.9706393666918, 117.21227761742348], [1583.3482983545491, 93.28860915227246], [1638.6735630485543, 103.50077790754942], [1644.2168949520724, 102.52500775914766], [1754.2999121728851, 97.17805174620413], [1770.779686401861, 117.79236877888135]]
#geo=[[0,32], [800,32], [900,38], [970,42], [1050, 40], [1180, 48], [1350, 60], [1390, 68], [1500, 72]]
#geo=Geo(geo)

# cadsd=CADSD(geo)
# spec=cadsd.get_ground_spektrum()


#cadsd=CADSD(geo)


#cadsd_balance(cadsd)
#print(cadsd.get_overblow_notes())
#df=cadsd.get_all_spektra_df()


# df2={
# 	"freq": [],
# 	"y": [],
# 	"spectrum": []
# }

# l=len(df)
# for s in ["impedance", "ground", "overblow"]:
# 	df2["freq"].extend(list(df.freq))
# 	df2["y"].extend(list(df[s]))
# 	df2["spectrum"].extend([s]*l)

# df2=pd.DataFrame(df2)

# sns.lineplot(data=df2, x="freq", y="y", hue="spectrum")
# plt.grid()  #just add this
# #g=plt.stackplot(list(df["freq"]),list(df["ground"]))
# #plt.ylim(0,)
# plt.show()
