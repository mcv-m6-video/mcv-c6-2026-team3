# C6 Team3

## Project structure
```
C6
├── config         <--- Configuration class for handling paths
├── detectors      <--- Detector classes for object detection from mask
├── models         <--- Models used for getting background removal
├── pipelines      <--- Complete pipeline for object detection 
├── plots          <--- Plotter notebook for slide plots
├── preprocess     <--- Preprocess functions for mask refinement
└── studies        <--- Studies programs for hyperparameter search
```
## Installation

Execute the following commands:

```
conda create -n env python=3.11 -y
conda activate env
pip install -r requirements.txt
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

## Usage (no ZBS)

In order to use our models you need to do:
```
python main.py -d data_folder -r result_folder
```
The data folder must contain the AICity dataset and the annotations file within it.

To change the model, you just need to open the main.py and change the model used.

If another preprocessing is needed, then change the preprocess_fn function within the main.py file as well.

The results will be a video with bounding boxes corresponding to the objects and another with the masks that are detected. In addition, a file called metrics.txt would also be found inside the folder, containing the mAP@5 results, as well as other metrics gotten from detectron2.

## ZBS Usage

The ZBS usage is different from the other models. You first need to compute the video mask using the ZBS model. In order to do that you can follow the instructions [here](https://github.com/CASIA-IVA-Lab/ZBS). Once the video with masks is computed, you can then execute
```
python eval_zbs.py -i mask_video -r result_folder -a annotation_file -d data_folder
```
As before, the data folder must contain the AICity dataset.

The results will be a video with bounding boxes corresponding to the objects. In addition, a file called metrics.txt would also be found inside the folder, containing the mAP@5 results, as well as other metrics gotten from detectron2.