import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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
    plt.title(station, loc='left')
    plt.title(t_str, loc='right')
    t_str =  datetime.strptime(t_str , '%Y/%m/%d-%H').strftime('%Y%m%d_%H')
    plt.savefig(f'{subexp_dir}/AE_{station}/{t_str}.png')
    plt.close()

# setting
subexp_dir = 'AE_station/'
# stations = [466880, 466900, 466910, 466920, 466930, 466940, 467050, 467060, 467080, 467571]
stations = [467270]
yr='20t22' #'18t21'

for station in stations:
    #load data
    pred_path = f'{subexp_dir}/AE_{station}/{station}_{yr}_bad.npy'  #!!!
    pred_Ts = np.load(pred_path)
    
    obs_path = f'../data/station_ds/{station}_H72_{yr}.h5' #!!!
    with h5py.File(obs_path, 'r') as f:
        obs_Ts = f['bad/temp'][:]
        obs_ts = f['bad/time'][:]
    
    # plot
    for i in range(len(obs_ts)):
        plot_temp(obs_Ts[i,:], pred_Ts[i,:], obs_ts[i])
    
