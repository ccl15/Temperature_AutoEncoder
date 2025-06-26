import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from matplotlib.patches import FancyArrowPatch

def get_sid_list(exp):
    namelist = f'{exp}/namelist.txt'
    with open(namelist, 'r') as f:
        lines = f.readlines()
        sids = [line.split()[0] for line in lines]
    return sids

def FN_Reveral(file):
    print(file)
    FN_rate = []
    Reversal_rate = []
    with h5py.File(file, 'r') as f:
        for sid in sids:
            Pmae = f[sid]['PassMAE'][:]
            Wmae = f[sid]['WarmMAE'][:]
            FNrate = sum(Pmae>th)/len(Pmae)*100
            Rrate = sum(Wmae<=th)/len(Wmae)*100 if len(Wmae)>0 else 110
            if FNrate >= 5:
                print(f'{sid} FN>5:{FNrate:.2f}, n:{len(Pmae)}')
            if Rrate <= 40:
                print(f'{sid} R<40:{Rrate:.2f}, n:{len(Wmae)}')

            FN_rate.append(FNrate)
            Reversal_rate.append(Rrate)
    return FN_rate, Reversal_rate

def pass_reversal(th):
    # 0: baseline; 1: transfer
    FN_0, Reversal_0 = FN_Reveral(f'{exp}/cases_mae.h5')
    FN_1, Reversal_1 = FN_Reveral(f'{exp}/trans_cases_mae.h5')
    
    # plot 
    plt.figure()
    plt.plot(FN_0, Reversal_0, 'o', c='#FF3D00', ms=8, alpha=0.5, mec='none', label='Pre-Train')
    plt.plot(FN_1, Reversal_1, 'o', c='#00C853', ms=8, alpha=0.5, mec='none', label='Transfer')

    for x0, y0, x1, y1 in zip(FN_0, Reversal_0, FN_1, Reversal_1):
        plt.plot([x0, x1], [y0, y1], c='b', alpha=0.3, linewidth=1)
    
    plt.hlines(y=100, xmin=-5, xmax=90, color='gray', linewidth=1)
    plt.vlines(x=0, ymin=-5, ymax=115, color='gray', linewidth=1)

    plt.title(f'Threshold {th}')
    plt.xlabel("False Negative Rate of Normal")
    plt.ylabel("Reversal Rate of Warning")
    plt.legend()
    plt.axis([-1,15,-5,115])
    #plt.axis('equal')
    plt.savefig(f'{exp}/Trans_PR{th}_rate.png', dpi=200,)
    #plt.close()


#%%
exp = 'Pre_M30_f32k3'

sids = get_sid_list(exp)
th = 0.1
pass_reversal(0.1)



