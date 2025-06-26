import numpy as np
import pandas as pd
from pathlib import Path

#%% converter
class ScodeConverter:
    def __init__(self):
        with open('module/StationCode.txt', 'r') as f:
            self.mapping = {line.split()[0]: line.split()[1] for line in f.readlines()}

    def __call__(self, scode):
        return self.mapping.get(scode, None)

#%% staion data  
def data_loader(folder):
    files = Path(folder).iterdir()

    all_rows = []
    for file in files:
        date0 = file.name[:10]

        with open(file, 'rb') as f:
            data = f.readlines()
        for line in data[::2]:
            parts = line.decode().strip().split()
            if len(parts) < 10: continue

            full_date = f'{date0} {parts[0]}'
            parts[9] = np.nan if parts[9] == '/' else float(parts[9])
            
            all_rows.append([full_date, parts[1], parts[9]])

    df = pd.DataFrame(all_rows, columns=['time', 'Scode', 'temp']).drop_duplicates()
    grouped = df.groupby('Scode')
    return grouped


def get_timeserise(dfg, period=16, time_interval='1min'):
    # Convert time column to datetime for time difference calculation
    dfg.sort_values(by='time').reset_index(drop=True)
    expected_diff = pd.Timedelta(time_interval)
    
    temps = dfg['temp'].values
    times = pd.to_datetime(dfg['time'], format='%Y-%m-%d %H%M').values

    ds_temp = []
    ds_time = []
    
    for i in range(period, len(dfg)+1):
        w_temp = temps[i-period:i]
        w_time = times[i-period:i]

        # Skip if any temperature value is NaN
        if np.isnan(w_temp).any():
            continue
            
        # Check time continuity
        time_diffs = np.diff(w_time)
        if np.all(time_diffs == expected_diff):
            ds_temp.append(w_temp)
            ds_time.append(dfg.iloc[i-1]['time'])
    
    ds_temp = np.array(ds_temp, dtype=np.float32)
    ds_time = np.array(ds_time)
    return ds_temp, ds_time