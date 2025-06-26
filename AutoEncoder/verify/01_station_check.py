import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py


#%% calculate 
def stn_high():
    df = pd.read_csv("/home/ccl/Resources/StnTown_Gau.txt", sep=r"\s+", encoding="big5", dtype=str)

    df["Elev"] = df["Elev"].astype(float)
    elevation_dict = dict(zip(df["StnID"], df["Elev"]))
    return elevation_dict

def noPass_station(th_list):
    stations = 0

    with open(f'{exp}/transfer.txt', 'w') as fout:
        fout.write(f'Sid High th{th_list}\n')

        with h5py.File(f'{exp}/cases_mae.h5', 'r') as f:
            for sid in sorted(f.keys()):
                Pmae = f[sid]['PassMAE'][:]
                
                # Pmae have any value over th
                cnt = []
                for th in th_list:
                    cnt.append( np.sum(Pmae > th) )
                
                if sum(cnt)>0:
                    s_high = elevation_dict[sid] if sid in elevation_dict else -1
                    cnt_str = " ".join(f'{n:3d}' for n in cnt)
                    fout.write(f'{sid}  {s_high:6.1f} {cnt_str} \n')
                    stations += 1
        print(stations)


def namelist(col, num):
    with open(f'{exp}/namelist.txt', 'w') as fout:
        with open(f'{exp}/transfer.txt', 'r') as fin:
            # skip lines
            lines = fin.readlines()[1:]
            for line in lines:
                part = line.split()
                if int(part[col]) > num:
                    fout.write(line)

#%% fig 
def pass_reversal(th):
    plt.figure()
    FN_rate = []
    Reversal_rate = []

    with h5py.File(f'{exp}/cases_mae.h5', 'r') as f:
        sids = f.keys()
        for sid in sids:
            Pmae = f[sid]['PassMAE'][:]
            Wmae = f[sid]['WarmMAE'][:]
            FNrate = sum(Pmae>th)/len(Pmae)*100
            Rrate = sum(Wmae<=th)/len(Wmae)*100 if len(Wmae)>0 else 110
            FN_rate.append(FNrate)
            Reversal_rate.append(Rrate)
            if FNrate>2:
                print(sid, f'{FNrate:.2f}')

    plt.plot(FN_rate, Reversal_rate, 'o', c='#FF3D00', ms=10, alpha=0.5, mec='none')
    plt.hlines(y=100, xmin=-5, xmax=90, color='gray', linewidth=1)
    plt.vlines(x=0, ymin=-5, ymax=115, color='gray', linewidth=1)

    plt.title(f'Threshold {th}')
    plt.xlabel("False Negative Rate of Normal")
    plt.ylabel("Reversal Rate of Warning")
    # x y tick ratio equal
    #plt.axis([-5,80,-5,115])
    xmax = (np.max(FN_rate)//5+1)*5
    plt.axis([-1,xmax,-5,115])
    #plt.axis('equal')
    plt.savefig(f'{exp}/PR{th}_rate.png', dpi=200)
    #plt.close()
    #return FN_rate, Reversal_rate, sids

def plot_error_hist(bins, ylog=True,zoonin=False):
    # data process
    PassMAE=[]
    WarmMAE=[]
    with h5py.File(f'{exp}/cases_mae.h5', 'r') as f:
        for sid in f.keys():
            PassMAE.extend(f[sid]['PassMAE'][:])
            WarmMAE.extend(f[sid]['WarmMAE'][:])
    # value over limit set to limit
    #PassMAE = np.clip(PassMAE, 0, bins[-1])
    #WarmMAE = np.clip(WarmMAE, 0, bins[-1])
    
    # plot
    plt.figure()
    plt.hist(PassMAE, bins=bins, alpha=0.7, label=f'Pass ({len(PassMAE)})')
    plt.hist(WarmMAE, bins=bins, alpha=0.7, label=f'Warning ({len(WarmMAE)})')
    plt.legend()
    plt.xlabel("Last 3 Temperature MAE")
    plt.ylabel("Count")
    
    if zoonin:
        plt.xlim(0, bins[-1])
    else:
        plt.xlim(0, 1)
    
    if ylog: 
        plt.yscale('log')
        plt.savefig(f'Hist_{exp}_ylog.png', dpi=200)
    else:
        plt.savefig(f'Hist_{exp}.png', dpi=200)

def plot_hight_noPass():
    high = []
    NP = []
    with open(f'{exp}/transfer.txt', 'r') as fin:
        lines = fin.readlines()[1:]
        for line in lines:
            part = line.split()
            if int(part[2])>3:
                high.append(float(part[1]))
                NP.append(int(part[2]))
    
    plt.axis([0,4000,0,40])
    plt.plot(high, NP,'.')
    plt.xlabel("Station Height")
    plt.ylabel("Number of NoPass Cases")
    plt.savefig(f'{exp}/High_NP.png', dpi=200)


#%% main -----------------------
exp_list = ['Pre_M30_f32k3']

for exp in exp_list:
    elevation_dict = stn_high()
    pass_reversal(0.1)

    #bins = np.arange(0, 10.1, 0.1)
    #plot_error_hist(bins,True)

    #noPass_station(th_list = [0.1, 0.2])
    #namelist(col=2, num=5)
