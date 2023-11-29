# calculate thershold of all station
import numpy as np
import h5py
import pandas as pd

#%% load data ------------------------------------
def _load_staion(sid, vers):
    exp_dir = 'AE_min_test'
    # load predict
    if vers == 0:
        pred_path = f'{exp_dir}/{sid}.npy'
    elif vers == 1:
        pred_path = f'{exp_dir}/{sid}_1.npy'
        
    pred = np.load(pred_path)
    obs = np.load(f'../data/7processed/{sid}.npz')['temp']
    return pred, obs

def _get_threshold(sid, n, vers, outlier=None):
    pred_g, obs_g = _load_staion(sid, vers)
    errors = np.array([np.mean(abs(yt[-n:]-yp[-n:])) for yt, yp in zip(obs_g, pred_g)])
    if outlier:
        errors = errors[errors<=outlier]    

    mean = np.mean(errors)
    std = np.std(errors)
    th = mean +3*std
    return mean, std, th

#%% output -----------------------
def create_all_staion_th_file():    
    for outlier in [0.1]:
        with open(f'threshold/th_16m3_{outlier}.txt', 'w') as fout:  # !!!!!
            fout.write('ID      mean   std    th\n')
            with open('../data/test_minute.txt', 'r') as fin:  #!!!!!!!
                # calculate threshold for each station
                for line in fin:
                    sid = line.strip().split()[0]
                    mean, std, th = _get_threshold(sid, 3, 0, outlier)
                    fout.write(f'{sid} {mean:5.4f} {std:5.4f} {th:5.4f}\n')


#%% compaire new threshold with old threshold
def compaire_new_old_th():
    # load old threshold
    th_list = {}
    with open('threshold/th_16m3.txt', 'r') as f:
        next(f) # skip first line
        for l in f:
            sid = l.split()[0]
            th_list[sid] = l.strip()


    # calculate new threshold
    with open(f'../data/list_re.txt', 'r') as f:
        for l in f:
            sid = l.strip()
            mean, std, th = _get_threshold(sid, n=3, vers=1)
            print('old',th_list[sid])
            print(f'new {sid} {mean:5.4f} {std:5.4f} {th:5.4f}\n')


def check_all_threshold(): 
    file_name = 'threshold/th_16m3.txt'
    df = pd.read_csv(file_name,  sep='\s+')

    print(df[df['mean']>0.2])
    print(df[df['std']>0.2])
    print(df[df['th']> 0.6])



#%% all stations pass ratio
def all_stations_pass_ratio():
    with open('threshold/passratio_16m3_0.1.txt','w') as fout:
        with open('threshold/th_16m3_0.1.txt', 'r') as f:
            next(f) # skip first line
            for l in f:
                tmp = l.strip().split()
                sid = tmp[0]
                th = tmp[3]
                pred_g, obs_g = _load_staion(sid, 0)
                errors = [np.mean(abs(yt[-3:]-yp[-3:])) for yt, yp in zip(obs_g, pred_g)]
                ratio = np.sum(np.array(errors) < float(th))/len(errors)
                fout.write(f'{sid} {ratio:.3f}\n')

#%% ---------------------------------------

#compaire_new_old_th()
#create_all_staion_th_file()
#check_all_threshold()
all_stations_pass_ratio()

