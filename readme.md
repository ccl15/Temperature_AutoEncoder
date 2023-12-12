# Temperature Autoencoder
Temperature anomaly detection in time series.


## Train Model
### Train sub-experiments by ymal
```
python main.py <experiment_path> -GPU <GPU_number>
```

### Train Multiple Stations from a Name File
Files needed:
- Template experiments ymal file
- name_file : list of stations

Edit the run.sh script and then execute it:
```
./run.sh
```

## Calculate thershold
**threshold.py** or **minute_threshold.py** at **output** folder


## Run autoencoder to test timeserise
At **operate** folder

Files needes:
- main code : AE_QC.py
- model : AE_2_0.py
- weight of stations : model/
- station threshold : th_24h3.py

Execute
```
python AE_QC.py <station_id> <datetime>
```
