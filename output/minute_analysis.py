import numpy as np
from pathlib import Path
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})

def MAE(yt, yp):
    return np.mean(abs(yt-yp))

subexp_dir = f'AE_minute/M221_lev2_f32_len16_2e4_p2p'

def one_hist(station):
    ds = np.load(f'../data/7processed/{station}.npz')
    T_obs = ds['temp']
    T_pred = np.load(f'{subexp_dir}/{station}.npy')
    #loss =[MAE(obs, pred) for obs, pred in zip(T_obs, T_pred)]
    loss3 =[MAE(obs[-3:], pred[-3:]) for obs, pred in zip(T_obs, T_pred)]

    plt.figure()
    kwargs = dict(alpha=0.5,  bins=50, range=(0,0.01), density=True, stacked=True)
    plt.hist(loss3, **kwargs, color='#2196F3')
    
    plt.xlim([0, 0.01])
    plt.ticklabel_format(axis='x', style='sci',scilimits=(0,0))
    plt.xlabel('MAE')
    plt.ylabel('Count')
    plt.savefig(f'fig_min/hist_3m_{station}.png', bbox_inches='tight', dpi = 200)

one_hist('46690')

def all_pdf():
    stations = Path(subexp_dir).glob('*.npy')

    plt.figure()
    for path1 in stations:
        station = path1.stem
        # load test data
        ds = np.load(f'../data/7processed/{station}.npz')
        T_obs = ds['temp']
        # load prediction
        T_pred = np.load(path1)

        # calculate loss 
        loss =[MAE(obs, pred) for obs, pred in zip(T_obs, T_pred)]
        # calculate pdf
        pdf = norm.pdf(loss, loc=np.mean(loss), scale=np.std(loss))
        # plot pdf
        plt.plot(loss, pdf) #, label=station)

    plt.xlabel('MAE')
    plt.ylabel('PDF')
    plt.axis([0,5,0,100])
    plt.yscale('log')
    # plt.legend()
    plt.savefig('fig_min/logpdf_all_stations_p2p.png', bbox_inches='tight', dpi = 200)
#all_pdf()
