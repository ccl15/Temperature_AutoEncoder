import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import h5py
import json

#%% calculate 
def stn_high(code=6):
    with open("/home/ccl/Resources/StnTown_Sta_20250520.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    elevation_dict = {item["ID"][:code]: item["Elev"] for item in data["Item"]}
    return elevation_dict

def FNR(th_list, namelist, out_i=None, out_th=None):
    # calculate
    FN_rate = []
    stations_high=[]
    for name in namelist:
        with h5py.File(f'{exp}/{name}.h5', 'r') as f: #!!!!!!!
            for sid in f.keys():
                if sid in elevation_dict:
                    s_high = elevation_dict[sid]
                else:
                    continue
                stations_high.append(s_high)
                
                Pmae = f[sid]['PassMAE'][:]
                sid_FN = []
                for th in th_list:
                    #FNrate = sum(Pmae>th)/len(Pmae)*100
                    FN = sum(Pmae>th)
                    sid_FN.append(FN)
                FN_rate.append(sid_FN)

                # check
                if out_i is not None and out_th is not None:
                    if sid_FN[out_i] > out_th:
                        print(sid, s_high, f'{sid_FN[out_i]}')

    FN_rate = np.array(FN_rate)
    return FN_rate, stations_high

#%% fig
def FN_high_scatter(i, vmax=None, tt='', fname='FN_Alt'): 
    # vs hight
    plt.figure()
    plt.scatter(stations_high, FN[:, i], s=40, edgecolors='none', alpha=0.7)
    #plt.plot([0,20],[2,2])
    plt.xlabel("Station Altitude (m)")
    #plt.ylabel("FN Rate (%)")
    plt.ylabel("Flase Negative Number")
    plt.title(f'{tt}th={th_list[i]}')
    #plt.gca().set_aspect('equal')
    if vmax is None:
        vmax = (max(FN[:,i])//50+1)*50
    plt.axis([0,4000,0,vmax])
    plt.savefig(f'{exp}/{fname}.png', dpi=200, bbox_inches='tight')
        
    #% hist    
def FNN_th_hist(fn, vmax=100, binW=1):
    fig, axes = plt.subplots(len(th_list), 1, figsize=(8, 2*len(th_list)), sharex=False)
    
    bins = np.arange(0, vmax+0.05, binW)
    for i in range(len(th_list)):
        axes[i].hist(np.clip(FN[:,i],0, vmax)
                    ,bins=bins, alpha=0.9)
        axes[i].text(vmax*0.8, 8, f'th={th_list[i]}', size=16)
        #set domain
        axes[i].set_xlim([0, vmax])
        axes[i].set_ylim([0,10])
    
    plt.subplots_adjust(hspace=0.01)
    fig.supxlabel('False Negative Number')
    fig.supylabel('Stations Count')
    plt.tight_layout()
    plt.savefig(f'{exp}/{fn}.png', dpi=200)
#%%
def plot_error_hist(bins, ylog=True):
    # data process
    PassMAE=[]
    WarmMAE=[]
    for name in namelist:
        with h5py.File(f'{exp}/{name}.h5', 'r') as f:
            for sid in f.keys():
                PassMAE.extend(f[sid]['PassMAE'][:])
                
                if 'WarmMAE' in f[sid]:
                    WarmMAE.extend(f[sid]['WarmMAE'][:])
    # value over limit set to limit
    PassMAE = np.clip(PassMAE, 0, bins[-1])
    WarmMAE = np.clip(WarmMAE, 0, bins[-1])
    
    # plot
    plt.figure()
    plt.hist(PassMAE, bins=bins, alpha=0.7, label=f'Pass')
    plt.hist(WarmMAE, bins=bins, alpha=0.7, label=f'Warning')
    plt.legend()
    plt.xlabel("Last 3 Temperature MAE")
    plt.ylabel("Count")
    plt.xlim(0, bins[-1])

    mean = np.mean(PassMAE)
    std = np.std(PassMAE)
    print(f'Pass: mean={mean:.2f}, std={std:.2f}')

    if ylog: 
        plt.yscale('log')
        plt.savefig(f'{exp}/Hist_ylog_train.png', dpi=200)
    else:
        plt.savefig(f'{exp}/Hist_train.png', dpi=200)


#%% main -----------------------
#elevation_dict = stn_high()
#th_list = [0.1, 0.2]
#%%

exp = 'AE_HR16/fY23'
namelist = ['y2023']
bins = np.arange(0, 1.0001, 0.01)
plot_error_hist(bins)

    #FN, stations_high =FNR(th_list, namelist) 
    #FN_high_scatter(1, 70, f'20{yr}, ', f'y{yr}_FNN_Alt')
    #FNN_th_hist('ThFnn_hist_y24')
'''
exp = 'AE_HR16/Trans_2023'
for yr in [22]:
    for dd in [15]:
        for mm in [15, 20]:
            namelist = [f'y20{yr}_d{dd}_m{mm}']
            FN, stations_high =FNR(th_list, namelist,0, 50)
            FN_high_scatter(1, 70 , f'20{yr}, d{dd}m{mm}, ', f'N_y{yr}_d{dd}m{mm}')
# 10 min

exp = 'AE_10min/Pre_M30_f32k3'
for yr in [24]:
    namelist = [f'cases_20{yr}']
    FN, stations_high =FNR(th_list, namelist) 
    FN_high_scatter(1, 60, f'20{yr}, pre-train, ', f'y{yr}_Alt')



exp = 'AE_10min/LH_fs'
namelist = ['y2024_m15']
FN, stations_high =FNR(th_list, namelist, 0, 20)
FN_high_scatter(1, 60, '2024, d15m15', 'Y24_d15m15_1em5')
'''
#for yr in [22,24]:
#    namelist = [f'y20{yr}_L15',f'y20{yr}_H15']
#    FN, stations_high =FNR(th_list, namelist,1,10)
    #FN_high_scatter(0, 80, f'20{yr}, d15, ', f'y{yr}_2P')

#%% 1min
'''
elevation_dict = stn_high(code=5)
th_list = [0.1,0.15]
for exp in ['AE_min/AE222_f32k3_1em4']:
    print(exp)
    namelist = ['Pre_2024']
    FN, stations_high =FNR(th_list, namelist, 0,20) 

    FN_high_scatter(0, 100, 'y24_pre ','zoom')
    FNN_th_hist('ThFnn_hist_y24_relu',200,5)

#bins = np.arange(0, 1.0001, 0.01)
#plot_error_hist(bins)


#
AE_min/AE222_f32k3_1em4
46686 32.0 30
46755 3844.8 49
A0Z08 3171.0 35
CAAH6 18.0 153
'''