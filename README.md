# Traffic-Light-Classification
Traffic Light detection and classification

## Dataset
Bosch Small Traffic Lights Dataset
* https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset
* https://github.com/bosch-ros-pkg/bstld


## Model
PyTorch FasterRCNN

## Trained Model URL
https://byu.box.com/s/4w3b8h4zwoq4gzwz8vhkl0us8w5chlq1


## Instructions
1. Download the dataset and follow the official instructions to extract the dataset
2. The extracted files will contain images and yaml files. 
3. Run the script `create_dataset_pkl.py` with a yaml file as argument.
4. The above script will create a pickle file in the same directory as the yaml file and `bstld_(yaml file name).pkl`
5. Run `main.py --mode train --dir <directory where pickle file exists>` to train the FasterRCNN model
6. Run `main.py --mode test --dir <directory where pickle file exists>` to test the trained FasterRCNN model

## Model test on an Image
![TF_detection](https://user-images.githubusercontent.com/1760420/101208526-5235da80-3640-11eb-89e5-0e67f5c28c03.png)
