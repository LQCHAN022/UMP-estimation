# Remote Sensing Landuse

## Setup

Create a conda enviroment:

```
conda env create -f environment.yml
```

If PyTorch does not run with GPU, or if the PyTorch library is the CPU only version:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
## File Directory
```
├── 12ch_training.py
        - The script used for training
├── data_pipeline.ipynb
        - Data pipeline to assemble data
├── sentinel_ump_12.ipynb
        - Building of Dataset and experimenting with training
├── results_visualisation.ipynb
        - Visualisation of results
├── utils -> Collection of utilities, including helper functions as well as model
│   ├── convert_gml_to_shp.py
│   ├── data_preprocessing.py
│   ├── data_utils.py
│   ├── gee_downloader.py
│   ├── gml_utils.py
│   ├── istarmap.py
│   ├── landsat_downloader.py
│   ├── models
│   │   ├── modules.py
│   │   ├── net.py
│   │   └── senet.py
│   └── sp_utils.py
```