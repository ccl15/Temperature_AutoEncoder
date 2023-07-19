"""
Singal station different hour and weight line plot.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})



def MAE(yt, yp):
    return np.mean(abs(yt-yp))

def _load_staion_both(sid):
    pred_path = f'{exp_dir}/AE_{sid}/{sid}_18t21_good.npy'
    pred_g = np.load(pred_path)
    pred_path = f'{exp_dir}/AE_{sid}/{sid}_18t21_bad.npy'
    pred_b = np.load(pred_path)
    
    obs_path = f'../data/2ds_46/{sid}_H24_18t21.h5'
    with h5py.File(obs_path, 'r') as f:
        obs_g = f['good/temp'][:]
        obs_b = f['bad/temp'][:]
        time_b = f['bad/time'][:]
    return pred_g, obs_g, pred_b, obs_b, time_b

def recall_rate(good, bad, weight):
    mean = np.mean(good)
    std = np.std(good)
    th = mean + weight * std
    r =  sum(bad <= th)/len(bad)
    return r


def recall_plot():
    weight = [3, 2.5, 2]
    # weight = [1,1.5]
    MAE_good = [ [MAE(pred_g[i][-n:], obs_g[i][-n:]) for i in range(len(obs_g))]  for n in range(1,11)] 
    MAE_bad  = [ [MAE(pred_b[i][-n:], obs_b[i][-n:]) for i in range(len(obs_b))]  for n in range(1,11)] 
    colors = ['#0D47A1', '#1E88E5', '#64B5F6']    
    for w in range(len(weight)):
        recall = [recall_rate(MAE_good[i], MAE_bad[i], weight[w]) for i in range(10)]
        plt.plot(recall, c=colors[w], label=f'+{weight[w]} std')
    
    plt.legend()
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.9))
    plt.title(sid)
    plt.ylabel('Recall rate')
    plt.xlabel('last n hours to calculate error')
    plt.xticks(range(10), range(1,11) ,fontsize = 10)
    plt.savefig('fig/recall_{sid}.png', bbox_inches='tight', dpi = 200)
    


exp_dir = 'AE_station_24'
sid = '466920'
pred_g, obs_g, pred_b, obs_b, time_b = _load_staion_both(sid)
recall_plot()