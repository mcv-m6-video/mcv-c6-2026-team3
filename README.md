# C6 Team3

## Installation

Execute the following commands:

```
conda create -n env python=3.11 -y
conda activate env
pip install -r requirements.txt
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

## Folder structure

The data needs to be inside a data folder. Extract zips directly in data folder for it to work (later will be replaced with argument and argparse)