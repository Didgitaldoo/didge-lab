from cad.cadsd.cadsd import CADSDResult
from cad.calc.geo import Geo

def smooth_geo(geo):

    pass

if __name__ == "__main__":
    
    geo=[[308.41040028593943, 34.483167621383984], [336.44770940284303, 38.86453430197376], [362.64491281677107, 48.273606300042985], [403.193440358051, 56.18950384638686], [420.74688803924005, 61.6066106436609], [429.92480929935044, 68.2165856518555], [445.9286892928244, 66.8828361152447], [510.43203892603094, 63.68007899450243], [544.0037643774868, 57.67723398268195], [560.4588648669693, 52.3450060636075], [608.8787286458708, 43.771971294100034], [615.4868867272795, 41.25428918150348], [625.2947725955073, 40.43498448769555], [677.0490290722233, 54.2261787183299], [708.0220198822772, 52.97647330310105], [725.3662474110508, 65.0492138251598], [744.5622899599838, 70.03095367964045], [794.827047814253, 72.18982182512977], [833.2109865282634, 61.31043030761378], [867.2204986057064, 66.37965084153542], [878.4201357612385, 49.810679799266396], [887.7745522091608, 66.7560034234089], [937.3047709378608, 56.3788497822419], [986.4294826802624, 60.90851551925686], [1219.5441518039586, 67.03967650536613], [1261.6789102606613, 88.61374321009784], [1281.554184459171, 90.22339636474396], [1289.7162193775648, 100.49453779893668], [1323.9338013825304, 121.04941696674678], [1349.8311120721955, 121.72811889287622], [1399.472650994982, 141.46702022118663], [1400.5065793854565, 135.8075510610561], [1410.8881975934898, 124.65919003218724], [1448.3747757098172, 121.8397299440968], [1490.5179695633967, 112.88351708541843], [1534.6250337773674, 83.46737310274192], [1594.3340748492624, 90.86597438115896], [1850.4624017156366, 123.16587751880633]]
    geo=Geo(geo=geo)

    cadsd=CADSDResult.from_geo(geo)
    print(cadsd.peaks)



    geo=[[308.41040028593943, 34.483167621383984], [336.44770940284303, 38.86453430197376], [362.64491281677107, 48.273606300042985], [403.193440358051, 56.18950384638686], [420.74688803924005, 61.6066106436609], [429.92480929935044, 68.2165856518555], [445.9286892928244, 66.8828361152447], [510.43203892603094, 63.68007899450243], [544.0037643774868, 57.67723398268195], [560.4588648669693, 52.3450060636075], [608.8787286458708, 43.771971294100034], [615.4868867272795, 41.25428918150348], [625.2947725955073, 40.43498448769555], [677.0490290722233, 54.2261787183299], [708.0220198822772, 52.97647330310105], [725.3662474110508, 65.0492138251598], [744.5622899599838, 70.03095367964045], [794.827047814253, 72.18982182512977], [833.2109865282634, 61.31043030761378], [867.2204986057064, 66.37965084153542], [878.4201357612385, 49.810679799266396], [887.7745522091608, 66.7560034234089], [937.3047709378608, 56.3788497822419], [986.4294826802624, 60.90851551925686], [1219.5441518039586, 67.03967650536613], [1261.6789102606613, 88.61374321009784], [1281.554184459171, 90.22339636474396], [1289.7162193775648, 100.49453779893668], [1323.9338013825304, 121.04941696674678], [1349.8311120721955, 121.72811889287622], [1399.472650994982, 141.46702022118663], [1400.5065793854565, 135.8075510610561], [1410.8881975934898, 124.65919003218724], [1448.3747757098172, 121.8397299440968], [1490.5179695633967, 112.88351708541843], [1534.6250337773674, 83.46737310274192], [1594.3340748492624, 90.86597438115896], [1850.4624017156366, 123.16587751880633]]
    x0=geo[0][0]
    for i in range(len(geo)):
        geo[i][0]-=x0
    geo=Geo(geo=geo)

    cadsd=CADSDResult.from_geo(geo)
    print(cadsd.peaks)
