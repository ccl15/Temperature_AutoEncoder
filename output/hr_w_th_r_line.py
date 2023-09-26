"""
Singal station different hour and weight line plot.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
from pathlib  import Path
from scipy.stats import rv_histogram


def MAE(yt, yp):
    return np.mean(abs(yt-yp))

def _load_staion_both(sid):
    pred_path = f'{exp_dir}/AE_{sid}/{sid}_good.npy'
    pred_g = np.load(pred_path)
    pred_path = f'{exp_dir}/AE_{sid}/{sid}_bad.npy'
    pred_b = np.load(pred_path)
    
    obs_path = f'../data/2ds/{sid}_H24_18t22.h5'
    with h5py.File(obs_path, 'r') as f:
        obs_g = f['good/temp'][:]
        obs_b = f['bad/temp'][:]
        time_b = f['bad/time'][:]
        code_b = f['bad/code'][:]
    # get index of code_b ==1 or code >10
    idx = np.where((code_b == 1) | (code_b > 10))
    return pred_g, obs_g, pred_b[idx], obs_b[idx], time_b[idx]

def recall_rate(good, bad, weight):
    mean = np.mean(good)
    std = np.std(good)
    th = mean + weight * std
    r =  sum(bad <= th)/len(bad)
    return r

def pass_and_all(good, bad, weight):
    mean = np.mean(good)
    std = np.std(good)
    th = mean + weight * std
    return sum(bad <= th), len(bad)





#%% good bad histogram
exp_dir = 'AE_station_24'
Pdirs = list(Path(exp_dir).glob('AE_46*'))
n = 3

Agood = []
Abad = []
for i in range(len(Pdirs)):
    sid = Pdirs[i].name[3:]
    pred_g, obs_g, pred_b, obs_b, time_b = _load_staion_both(sid)
    MAEg = [MAE(pred_g[i][-n:], obs_g[i][-n:]) for i in range(len(obs_g))]
    MAEb = [MAE(pred_b[i][-n:], obs_b[i][-n:]) for i in range(len(obs_b))]
    Agood.extend(MAEg)
    Abad.extend(MAEb)
print('load done')


kwargs = dict(alpha=0.5,  bins=20000, range=(0,200), density=True, stacked=True)
plt.hist(Agood, **kwargs, color='#2196F3', label='pass')
plt.hist(Abad,  **kwargs, color='#E64A19',label='alarm')

plt.xlabel('Last 3 hours MAE')
plt.ylabel('Probability density')
plt.yscale('log')
# plt.axis([-0.1, 1.5, 0, 2])
plt.xlim([-0.1,2])
# plt.title(sid)
mean = np.mean(MAEg)
std = np.std(MAEg)
# plt.plot(mean, 0.01, c='#388E3C', ms=20)
# plt.plot(mean+3*std, 0.01, c='#AFB42B', ms=20)
plt.legend()
plt.savefig('fig/histlog_gb_A46.png',bbox_inches='tight', dpi = 200)

#%% weight recall

exp_dir = 'AE_station_24'
Pdirs = list(Path(exp_dir).glob('AE_46*'))
n = 3

weight = [3, 2.5, 2]
ds = np.zeros((len(Pdirs), len(weight), 10, 2))

num = 0
for i in range(len(Pdirs)):
    sid = Pdirs[i].name[3:]
    print(sid)
    pred_g, obs_g, pred_b, obs_b, time_b = _load_staion_both(sid)
    num += len(time_b)
    MAEg = [ [MAE(pred_g[i][-n:], obs_g[i][-n:]) for i in range(len(obs_g))]  for n in range(1,11)] 
    MAEb  = [ [MAE(pred_b[i][-n:], obs_b[i][-n:]) for i in range(len(obs_b))]  for n in range(1,11)] 
    for w in range(len(weight)):
        for n in range(10):
            ds[i,w,n,:] = pass_and_all(MAEg[n], MAEb[n], weight[w])

#%%
def recall_plot():
    colors = ['#01579B', '#388E3C','#F9A825']    
    for w in range(len(weight)):
        recall = [sum(ds[:,w,n,0])/sum(ds[:,w,n,1])for n in range(10)]
        plt.plot(recall, c=colors[w], label=f'+{weight[w]} std')
    
    # plt.legend()
    plt.legend(loc='right', bbox_to_anchor=(1, 0.45))
    plt.title( f'{num} cases', loc='right', fontsize = 12)
    plt.ylabel('Recall rate' ,fontsize = 16)
    plt.xlabel('QC last N hours' ,fontsize = 16)
    plt.xticks(range(10), range(1,11))
    # plt.ylim([0.7,0.82])
    plt.savefig('fig/recall_all46.png', bbox_inches='tight', dpi = 200)    
        
recall_plot()
