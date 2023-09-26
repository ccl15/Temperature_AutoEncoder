import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


# font_setting
s, m, l = 10, 14, 16
plt.rc('font', size= m)
plt.rc('figure', titlesize= l)
plt.rc('xtick', labelsize= s)
plt.rc('ytick', labelsize= s)


def plot_temp(T_obs, T_pred, t0):
    plt.figure()
    x = range(-len(T_obs)+1, 1, 1)
    plt.plot(x, T_obs, c='b', label='obs. temp.')
    plt.plot(x, T_pred,  c='r', label='AE output')
    
    t_str = t0.decode('utf-8')
    plt.legend()
    plt.ylabel('Temperature (oC)')
    plt.xlabel('Time (hr)')
    plt.title(sid, loc='left')
    plt.title(t_str, loc='right')
    t_str =  datetime.strptime(t_str , '%Y/%m/%d-%H').strftime('%Y%m%d_%H')
    plt.savefig(f'{subexp_dir}/timeseries/{sid}_{t_str}.png')
    plt.close()


def plot_temp_2(T_obs, T_pred, t_str):
    #x = range(-len(T_obs)+1, 1, 1)
    x = range(-15, 1, 1)
    for i , dt in enumerate([-0.5,-1]):
        plt.figure()
        plt.plot(x, T_obs[i,:], c='b', label='obs. temp.')
        plt.plot(x, T_pred[i,:],  c='r', label='AE output')

        plt.legend()
        plt.ylabel('Temperature (oC)')
        plt.xlabel('Time (hr)')
        plt.title(station, loc='left')

        plt.title(t_str, loc='right')
        #t_str =  datetime.strptime(t_str , '%Y/%m/%d-%H').strftime('%Y%m%d_%H')
        plt.savefig(f'{subexp_dir}/{station}_{i}_{t_str}.png')
        plt.close()

#%% For minute
exps = [
    'M221_lev2_f32_len16_2e4_p2p',
    ]


for subexp in exps:
    subexp_dir = f'AE_minute/{subexp}'
    station = '466900'
    # load data
    ds = np.load(f'{subexp_dir}/{station}_bad.npz')
    #obs = ds['obs']
    #pred = ds['pred']
    time = ds['date']
    bad_pred = np.array(ds['pred'])
    bad_obs  = np.array(ds['obs'])
    for i in range(len(bad_obs)):
        plot_temp_2(bad_obs[:,i,:], bad_pred[:,i,:], time[i])

#%% For hourly
'''
subexp_dir = 'AE_station_24'
Pdirs = list(Path(subexp_dir).glob('AE_46*'))
yr='18t22'

for pdir in Pdirs:
    sid = pdir.name[3:]
    #load data    
    obs_path = f'../data/2ds/{sid}_H24_{yr}.h5' #!!!
    with h5py.File(obs_path, 'r') as f:
        code_b = f['bad/code'][:]
        idx = np.where((code_b == 1) | (code_b > 10))
        obs_Ts = f['bad/temp'][idx]
        obs_ts = f['bad/time'][idx]
    
    pred_path = f'{subexp_dir}/AE_{sid}/{sid}_bad.npy'  #!!!
    pred_Ts = np.load(pred_path)[idx]
    
    # plot
    print(sid, len(obs_Ts))
    for i in range(len(obs_Ts)):
        plot_temp(obs_Ts[i,:], pred_Ts[i,:], obs_ts[i])

    
'''
