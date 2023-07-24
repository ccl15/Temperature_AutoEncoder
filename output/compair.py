import numpy as np
import pandas as pd
from itertools import chain
import matplotlib.pyplot as plt
from scipy import stats
import h5py
# from datetime import datetime


yr = '18t21'

#%%
def MAE(yt, yp):
    return np.mean(abs(yt-yp))

def MSE(yt, yp):
    return np.mean((yt-yp)**2)

def t_test(good, bad):
    test = []
    for i in range(len(good)):
        t, p = stats.ttest_ind(good[i], bad[i])
        if p<0.05:
            test.append(i)
    return test
   

#%% 

class StationVsArea():
    def __init__(self, sta_list, An, n):
        # load data and calculate error
        singal_bad = []
        singal_good = []
        area_bad = []
        area_good = []
        obs = []
        times = []
        pred_s = []
        pred_a = []
        self.n=n
        self.sta_list = sta_list
        self.An = An
        self.th_weight = 3
        
        # load single model predict
        for station in sta_list:
            subexp_dir = f'AE_station_24/AE_{station}' #!!!
            # subexp_dir = f'AE_2conv_0/H72_{station}'
            g_pred, g_obs, b_pred, b_obs, b_time = self._load_data(subexp_dir, station)
            times.append(b_time)
            obs.append(b_obs)
            pred_s.append(b_pred)
            singal_good.append([MAE(yt[-self.n:], yp[-self.n:]) for yp, yt in zip(g_pred, g_obs)])
            singal_bad.append([MAE(yt[-self.n:], yp[-self.n:]) for yp, yt in zip(b_pred, b_obs)])
        # load area model predict
        for station in sta_list:
            if An == 'A2':
                subexp_dir = 'AE_2conv_3A/T15_f12_1em4'
            elif An == 'A3':
                subexp_dir = 'AE_2conv_3A/T10_f12_1em4'
            elif An == 'north':
                subexp_dir = 'AE_2conv_3A/lat2454_f12_1em4'
            else:
                continue
            g_pred, g_obs, b_pred, b_obs, b_time  = self._load_data(subexp_dir, station)
            pred_a.append(b_pred)
            area_good.append([MAE(yt[-self.n:], yp[-self.n:]) for yp, yt in zip(g_pred, g_obs)])
            area_bad.append([MAE(yt[-self.n:], yp[-self.n:]) for yp, yt in zip(b_pred, b_obs)])
        
        # push to self
        self.singal_bad = singal_bad
        self.singal_good = singal_good
        self.area_bad = area_bad
        self.area_good = area_good
        self.obs = obs
        self.pred_s = pred_s
        self.pred_a = pred_a
        self.times = times
    
    def _load_data(self, subexp_dir, station):
        pred_path = f'{subexp_dir}/{station}_{yr}_good.npy'
        g_pred = np.load(pred_path)
        pred_path = f'{subexp_dir}/{station}_{yr}_bad.npy'
        b_pred = np.load(pred_path)
        
        obs_path = f'../data/station_ds/{station}_H72_{yr}.h5'
        with h5py.File(obs_path, 'r') as f:
            g_obs = f['good/temp'][:]
            b_obs = f['bad/temp'][:]
            b_time = f['bad/time'][:]
        return g_pred, g_obs, b_pred, b_obs, b_time

    def check_pass_t_test(self):
        for i in range(len(self.sta_list)):
            t, p = stats.ttest_ind(self.singal_good[i], self.singal_bad[i])
            if p < 0.05:
                print('Sta pass')
            else:
                print('Sta not pass', self.sta_list[i])
            t, p = stats.ttest_ind(self.area_good[i], self.area_bad[i])
            if p < 0.05:
                print('Area pass')
            else:
                print('Area not pass', self.sta_list[i])
                
    ## plot error scatter -------------------------------------------------------------
    def plot_error_scatter(self):
        for i in range(len(self.sta_list)):
            plt.figure(figsize=(5,4))
            plt.scatter(self.area_good[i], self.singal_good[i], s=2, c='#00838F', alpha=0.7, label=' Pass')
            plt.scatter(self.area_bad[i],  self.singal_bad[i],  s=5, c='#EF6C00', alpha=1, label=' Alarm')

            plt.legend()
            plt.xlabel('Area model',fontsize = 12)
            plt.ylabel('Station model', fontsize = 12)
            plt.title(f'{self.sta_list[i]} MAE', fontsize = 14)
            # set  apsect equal
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis([0,0.5,0,0.5])
            plt.savefig(f'fig/SvsA_{self.An}_{self.sta_list[i]}.png', bbox_inches='tight', dpi=200)
            plt.close()
    

    ## plot error box -------------------------------------------------------------
    def _set_box_color(self, bp, color, barname):
        for element in ['boxes', 'whiskers', 'fliers', 'caps', 'medians']:
            plt.setp(bp[element], color = color, linewidth=1.5)
        plt.plot([], c=color, label=barname)

    def plot_error_box(self):
        sl = len(self.sta_list)
        plt.figure(figsize=(len(self.sta_list),4))

        flierprops = dict(marker=',', markerfacecolor='gray', markersize=4, markeredgecolor='none')

        B1 = plt.boxplot(self.singal_good, positions = np.arange(sl)-0.08, widths = 0.1, flierprops=flierprops)
        self._set_box_color(B1, '#0D47A1', 'Station Pass')
        B2 = plt.boxplot(self.singal_bad, positions = np.arange(sl)+0.08, widths = 0.1, flierprops=flierprops)
        self._set_box_color(B2, '#E65100', 'Station Alarm')
        # B3 = plt.boxplot(self.area_good, positions = np.arange(sl)+0.08, widths = 0.1, showfliers = False)
        # self._set_box_color(B3, '#42A5F5', 'Area Pass')
        # B4 = plt.boxplot(self.area_bad, positions = np.arange(sl)+0.24, widths = 0.1, showfliers = False)
        # self._set_box_color(B4, '#FF9800', 'Area Alarm')

        plt.xticks(np.arange(sl), self.sta_list, fontsize = 10)
        plt.legend( loc='upper right')# bbox_to_anchor=(1.4, 1),
        plt.xlim([-0.5, sl-0.5])
        #plt.ylim([-0.05, 0.7])
        # plt.xlabel('Station',fontsize = 14)
        plt.ylabel(f'Last {self.n} hours MAE', fontsize = 12)
        plt.savefig(f'fig/SvA_{self.An}_box.png', bbox_inches='tight', dpi=200)
        # plt.close()
  
    ## calculate confusion matrix ----------------------------------------------
    def _pass_threshold(self, good, bad):
        mean = np.mean(good)
        std = np.std(good)
        th = mean + self.th_weight * std
        result = np.where(bad > th, 1, 0)
        return th, result

    def _cm_calculate(self, ys, ya, th):
        # conver predict data to 0, 1
        pca = pd.Categorical(ya, categories=[0,1])
        pcs = pd.Categorical(ys, categories=[0,1])
        cm = pd.crosstab(pca, pcs, rownames=['Area'], colnames=['Station'])
        cm = cm.sort_index(ascending=False).sort_index(axis=1, ascending=False)
        return cm.values

    def confusion_matrix(self):
        lim = 0.8
        area_good_all = np.array(list(chain(*self.area_good)))
        for i in range(len(self.sta_list)):
            th_s, ys = self._pass_threshold(self.singal_good[i], self.singal_bad[i])
            th_a, ya = self._pass_threshold(area_good_all, self.area_bad[i])
            matrix = self._cm_calculate(ys, ya, th_s)
            
            
            plt.figure(figsize=(3,3))
            plt.scatter(self.area_bad[i],  self.singal_bad[i],  s=4, c='#EF6C00', alpha=1)

            plt.xlabel(f'Area th = {th_a:.2f}',fontsize = 12)
            plt.ylabel(f'Station th = {th_s:.2f}', fontsize = 12)
            plt.axvline(x = th_a, c='k', lw = 1)
            plt.axhline(y = th_s, c='k', lw = 1)
            try:
                plt.text(0.02, 0.02, matrix[1][1], fontsize = 14)
                plt.text(0.02, lim-0.1, matrix[1][0], fontsize = 14)
                plt.text(lim-0.1, 0.02, matrix[0][1], fontsize = 14)
                plt.text(lim-0.1, lim-0.1, matrix[0][0], fontsize = 14)
            except IndexError:
                pass
            plt.title(f'{self.sta_list[i]} \u03BC+{self.th_weight}\u03C3', fontsize = 14)
            # set  apsect equal
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis([0,lim,0,lim])
            plt.savefig(f'fig/SvsA_matrix_{self.An}_{self.sta_list[i]}.png', bbox_inches='tight', dpi=200)
            plt.close()
    
    ## plot time serise of Tturh, area, and station --------------------------------
    def plot_time_series(self):
        x = range(-71, 1, 1)
        area_good_all = np.array(list(chain(*self.area_good)))
        for i in range(len(self.sta_list)):
            # find traget index
            th_s, ys = self._pass_threshold(self.singal_good[i], self.singal_bad[i])
            th_a, ya = self._pass_threshold(area_good_all, self.area_bad[i])
            indexs = np.where((ys + ya) == 1)[0]
            # print(indexs)
            
            for j in indexs:
                plt.figure(figsize=(3,3))
                plt.plot(x, self.obs[i][j], c='k', label='Observed')
                plt.plot(x, self.pred_s[i][j], c='#E53935', label='Stat pred.')
                plt.plot(x, self.pred_a[i][j], c='#9CCC65', label='Area pred.')
                
                plt.legend()
                plt.xlim([-30,1])
                t_str = self.times[i][j].decode('utf-8')
                plt.title(f'{self.sta_list[i]} {t_str} \n MAE S:{self.singal_bad[i][j]:.2f}, A:{self.area_bad[i][j]:.2f}'
                          , fontsize = 12)
                plt.xlabel('Time', fontsize = 10)
                plt.ylabel('Temperature', fontsize = 10)
                plt.savefig(f'fig/times/SvsA_time_{self.An}_{self.sta_list[i]}_{j}.png', bbox_inches='tight', dpi=200)
                plt.tight_layout()
                plt.close()


sta_list_dict = {
    'A3': [467550, 467530],
    'A2': [466910, 466930, 467650, 467990],
    'north': [466880, 466900, 466910, 466920, 466930, 466940, 467050, 467060, 467080, 467571],
                 }
for An in ['north']:
    sta_area = StationVsArea(sta_list_dict[An], 'N24', 3)
    # sta_area.check_pass_t_test()
    # sta_area.plot_error_scatter()
    sta_area.plot_error_box()
    # sta_area.plot_time_series()
    # sta_area.confusion_matrix()
    
'''

for n in range(1,10):
    sta_area = StationVsArea([467990], 'A2', n)
    sta_area.check_pass_t_test()
'''

