# Materials Microscopic Image Segmentation
This is a PyTorch implementation of training and evaluation of U-Net deep 
learning model for semantic segmentation of materials microscopic images. The 
U-Net model proves to be able to achieve good performance for various materials
image datasets. Some highlights of the implementation are
- Consistent format for semantic segmentation dataset
- Hyperparameter configuration
- Data augmentation visualization

## Preparation
### Environment Configuration
The PyTorch and other necessary packages can be installed with pip. To do so,
run  
`pip install -r requirements.txt`

### Dataset Arrangement
The dataset folder need to be arranged in the following form:  
```
data
|--DATASET
    |--images
        a.tif (or a.png, a.jpg)
    |--labels
        a.npy
    |--labels_colored
        a.png
    |--splits
        split_cv.csv
        train.txt
```
The `labels` folder contains labels in numpy array form with data type 
`np.int8`. The shape of the array should be two dimension: (width, height) and 
each entry in the array gives the class of the pixel. Users can use image 
reading tools such as pillow or matplotlib to load image into a numpy array and 
use `np.save` to save it in `.npy` form. This process is to make sure that
labels take the same format for different dataset considering the diverse 
original file format of labels. Note that the name of the label numpy file is 
the same as the name of the image (`a.tif`, `a.npy`). The images in 
`labels_colored` folder show the labels in image format and are used only for 
visualization.

The files inside `splits` folder are used to split the dataset into train,
validation and test set for model development. There are two format of split files:
- `.csv`: Contains `name` and `split` columns. The `split` column specifies the
data split index the image belongs. This split format is especially designed for
cross-validation. See `./data/uhcs/splits/split_cv.csv` for an example.
- `.text`: Contains a list of image names in the dataset split. See 
`./data/uhcs/splits/train16A.txt` for an example.

## Training
We use `yaml` files to store configurations for datasets and model development. 
A `default.yaml` file specifying the default configuration should be created 
under `./configs/DATASET` for a new segmentation dataset. As an example, 
the contents of `./configs/uhcs/default.yaml` are
```
# dataset basics
dataset: uhcs                               # dataset name
img_folder: images                          # folder containing the images
label_folder: labels                        # folder containing the labels
n_classes: 4                                # number of classes                                

# train, validation and test split
split_info:
  type: CSVSplit                            # type of dataset split (CSVSplit, TextSplit)
  test_type: CSVSplit                       # type of test split (CSVSplit, TextSplit, validation, folder)
  split_file: split_cv.csv                  # name of split file
  split_col_name: split                     # split column name of the csv split file
  val_split_num: 1                          # split index for validation set (csv split only)
  test_split_num: 0                         # split index for test set (csv split only)
  train_reverse: True                       # if training set contains the remaining images
cross_validation:
  val_splits: [1, 2, 3, 4, 5, 0]            # validation split indices for k-fold cross-validation
  test_splits: [0, 1, 2, 3, 4, 5]           # test split indices for k-fold cross-validation

# data information
train_size: [224, 224]                      # size of cropped training images
eval_size: [484, 645]                       # size of evaluation images
mean: [0.485, 0.456, 0.406]                 # mean value in data normalization (ImageNet mean here)
std: [0.229, 0.224, 0.225]                  # standard deviation in data normalization (ImageNet std here)

# data augmentation
augmentations:                              # each line is a (operation: parameters) pair in the Albumentation package
  Flip: {p: 0.5}
  ShiftScaleRotate: {shift_limit: 0, scale_limit: [0,1], rotate_limit: 45, p: 0.8}
  RandomBrightnessContrast: {brightness_limit: 0.2, contrast_limit: 0.3, p: 0.8}

# training
n_epochs: 150                               # number of epochs
train_repeat: 8                             # number of times to repeat training set per epoch
loss_type: CE                               # type of loss function (CE, Dice, Jaccard)
ignore_index: -1                            # index to ignore in labels for computing loss
batch_size: 4                               # size of training batches
optimizer:
  type: AdamW                               # type of optimizer (AdamW, Adam, SGD)
  encoder_lr: 5.e-5                         # learning rate for U-Net encoder
  decoder_lr: 5.e-4                         # learning rate for U-Net decoder
  weight_decay: 1.e-2                       # weight decay (L2 penalty)
lr_scheduler:
  type: MultiStepLR                         # type of learning rate Scheduler (MultiStepLR, CAWR, CyclicLR, OneCycleLR)
  params: {milestones: [100], gamma: 0.3}   # parameters of the learning rate Scheduler
metric: mIoU                                # validation metric to optimize for model selection (mIoU, IoU_i)
```
To run experiments with customized configurations, create a `CONFIG.yaml` 
file specifying the different configuration parameters. For example, 
`example_textsplit.yaml` and `example_testfolder.yaml` under `./configs/uhcs`
give two other common scenarios for dataset splits.

To train a model, run `train.py` with the following argument:  
`python train.py --dataset DATASET --config CONFIG.yaml --gpu_id GPU_ID --seed SEED`

- `dataset`: name of the dataset (default: `uhcs`)
- `config`: configuration file (default: `default.yaml`)
- `gpu_id`: which GPU to run the script, cpu will be used if GPU is unavailable (default: `0`)
- `seed`: random seed to reproduce experiments (default: `42`)

During training, the model with the best metric score on validation set will be 
saved in `./checkpoints/DATASET/CONFIG/model.pth`.

## Evaluation
To evaluation the model trained by `CONFIG.yaml` on validation and test set, run
`eval.py` with the following argument:  
`python eval.py --dataset DATASET --config CONFIG.yaml --gpu_id GPU_ID --mode MODE`
- mode: `val` or `test` specifying validation or test

Evaluation will save the predictions in `./checkpoints/DATASET/CONFIG/predictions`.

## Colab Support
To train and evaluate models using GPU resources on Google Colab, please go to the notebook
`colab.ipynb`.

## Related Papers
DeCost, B., Lei, B., Francis, T., & Holm, E. (2019). High Throughput Quantitative 
Metallography for Complex Microstructures Using Deep Learning: A Case Study in Ultrahigh 
Carbon Steel. Microscopy and Microanalysis, 25(1), 21-29. https://doi.org/10.1017/S1431927618015635
arxiv: https://arxiv.org/abs/1805.08693

Durmaz, A.R., Müller, M., Lei, B. et al. A deep learning approach for complex 
microstructure inference. Nat Commun 12, 6272 (2021). https://doi.org/10.1038/s41467-021-26565-5
arxiv: 
