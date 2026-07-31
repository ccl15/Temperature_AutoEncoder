import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.patches import FancyArrowPatch


def FN_Reveral(file1, file2, th=0.1):
    FN1 = []
    FN2 = []
    sids = []
    with h5py.File(file1, 'r') as f1:
        with h5py.File(file2, 'r') as f2:
            for sid in f1.keys():
                if sid in f2.keys():
                    Pmae1 = f1[sid]['PassMAE'][:]
                    Pmae2 = f2[sid]['PassMAE'][:]

                    if sid not in ['46755', 'CAAH6','CAW01']:
                        FN1.append(sum(Pmae1>th))
                        FN2.append(sum(Pmae2>th))
                        sids.append(sid)
                    else:
                        print(sid, sum(Pmae1>th), sum(Pmae2>th))

    return FN1, FN2, sids



#%%
file1 = 'AE_min/TempAE_y2024.h5'
file2 = 'AE_min/AE222_f32k3_relu_1em4/Pre_2024.h5'
FN1, FN2, sids = FN_Reveral(file1, file2, 0.1)

#%%
plt.figure()
plt.plot([0,1.5], [0,1.5], alpha=0.5, c='gray', linewidth=1)
plt.scatter(FN1, FN2, s=10, alpha=0.8)

plt.xlabel("T AE")
plt.ylabel("T+RH AE")

plt.title("2024 Stational FNR")
plt.gca().set_aspect('equal')
#plt.axis([0,1.6,0,0.8])

#plt.savefig('AE_min/2024.png', dpi=200, bbox_inches='tight')


#%%
'''
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
'''