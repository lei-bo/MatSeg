# Materials Microscopic Image Segmentation
This is a PyTorch implementation of training and evaluation of popular models on  materials microscopic images datasets.
## Preparation
### Environment Configuration
- `pip install -r requirements.txt`
### Dataset Arrangement
```
dataset-folder
|──train
   |--images
      |--a.png
   |--labels
      |--a.png
   |--labels_npy
      |--a.npy
|--validate
   |--images
   |--labels
   |--labels_npy
|--test
   |--images
   |--labels
   |--labels_npy
   
```
## Training
Before train a model, edit `config.py` with desired parameters.
To train a model, run `main.py` with the following argument:  
`python3 main.py dataset train version`
- dataset: the name of the dataset folder 
- version: the configuration version (eg: `v1`)

However, this setting does not save the model. To save the model during training,
run  
`python3 main.py dataset train version --save`
## Evaluation
To evaluation a model on test images, run `main.py` with the following argument:  
`python3 main.py dataset evaluate version --test-folder TEST_FOLDER`
- dataset: the name of the dataset folder 
- version: the configuration version (eg: `v1`)
- TEST_FOLDER: the name of the test folder, default: `test`