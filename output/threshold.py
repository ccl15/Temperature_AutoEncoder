# calculate thershold of all station
import numpy as np
import h5py
import pandas as pd

#%% load data ------------------------------------
def _load_staion_pass(sid, vers):
    exp_dir = 'AE_station_24'
    #yr = '20t22' if sid == '467270' else '18t21'
    yr = '18t21'
    # load predict
    if vers == 0:
        pred_path = f'{exp_dir}/AE_{sid}/{sid}_18t21_good.npy'
    elif vers == 1:
        pred_path = f'{exp_dir}/AE_{sid}/{sid}_18t21_good_1.npy'
        
    pred_g = np.load(pred_path)
    # load truth
    obs_path = f'../data/2ds_46/{sid}_H24_{yr}.h5'  #!!!
    with h5py.File(obs_path, 'r') as f:
        obs_g = f['good/temp'][:]
        if len(pred_g) != len(obs_g):
            print(sid, 'data did not fit!')
    return pred_g, obs_g

def _load_staion_both(sid):
    pred_path = f'{exp_dir}/{sid}_18t21_good.npy'
    pred_g = np.load(pred_path)
    pred_path = f'{exp_dir}/{sid}_18t21_bad.npy'
    pred_b = np.load(pred_path)
    
    obs_path = f'../data/station_ds/{sid}_H24_18t21.h5'
    with h5py.File(obs_path, 'r') as f:
        obs_g = f['good/temp'][:]
        obs_b = f['bad/temp'][:]
        time_b = f['bad/time'][:]
    return pred_g, obs_g, pred_b, obs_b, time_b

def _get_threshold(sid, n, vers):
    pred_g, obs_g = _load_staion_pass(sid, vers)
    errors = [np.mean(abs(yt[-n:]-yp[-n:])) for yt, yp in zip(obs_g, pred_g)]
    mean = np.mean(errors)
    std = np.std(errors)
    th = mean +3*std
    return mean, std, th

#%% output -----------------------
def create_all_staion_th_file():    
    with open(f'threshold/threshold_h24_46_3.txt', 'w') as fout:  # !!!!!
        fout.write('ID      mean   std    th\n')
        with open('../data/list_46.txt', 'r') as fin:  #!!!!!!!
            # skip first line of fin
            #fin.readline()
            # calculate threshold for each station
            for line in fin:
                sid = line.strip()
                mean, std, th = _get_threshold(sid, n=3, vers=0)
                fout.write(f'{sid} {mean:5.3f} {std:5.3f} {th:5.3f}\n')

create_all_staion_th_file()

#%% compaire new threshold with old threshold
def compaire_new_old_th():
    # load old threshold
    th_list = {}
    with open('threshold/threshold_h24_C0_3.txt', 'r') as f:
        next(f) # skip first line
        for l in f:
            sid = l.split()[0]
            th_list[sid] = l.strip()


    # calculate new threshold
    with open(f'../data/list_C0_rerun.txt', 'r') as f:
        for l in f:
            sid = l.strip()
            mean, std, th = _get_threshold(sid, n=3, vers=1)
            print(th_list[sid])
            print(f'{sid} {mean:5.3f} {std:5.3f} {th:5.3f}')
            print('\n')

#compaire_new_old_th()

#%% ---------------------------------------
# check all threshold 
'''
file_name = 'threshold/threshold_h24_3.txt'
df = pd.read_csv(file_name,  sep='\s+')

print(df[df['mean']>0.3])
print(df[df['th']> 0.6])
'''
#%% -------------------------------------------------------------------------------------

def one_station_th_and_noPass(sid, n):
    with open(f'AE_station_24/threshold/th{n}_3std_{sid}.txt', 'w') as fout:
        pred_g, obs_g, pred_b, obs_b, time_b = _load_staion_both(sid)
        errors = [np.mean(abs(yt[-n:]-yp[-n:])) for yt, yp in zip(obs_g, pred_g)]
        # get th
        mean = np.mean(errors)
        std = np.std(errors)
        th = mean + 3*std
        
        # get time which MAE > th
        case = len(time_b)
        fout.write('ID      mean   std    th cases\n')
        fout.write(f'{sid} {mean:5.3f} {std:5.3f} {th:5.3f} {case}\n')
        fout.write('-------------------------------------\n')
        num = 0 
        for i in range(case):
            e = np.mean(abs(pred_b[i][-n:]-obs_b[i][-n:]))
            if e > th:
                ts = time_b[i].decode('utf-8')
                fout.write(f'{ts} {e:5.3f}\n')
                num += 1
        recall = f'{1-num/case:4.2f}'
        fout.write(f'recall: {recall}\n')
        print(sid, recall)
                

#for station in stations:
#    exp_dir = f'AE_station_24/AE_{station}'
#    one_station_th_and_noPass(station, n=3)
