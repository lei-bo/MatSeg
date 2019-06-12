# Materials Microscopic Image Segmentation
This is a PyTorch implementation of training and evaluation of popular models
on materials microscopic images datasets.
## Preparation
### Environment Configuration
The PyTorch and other necessary packages can be installed with pip. To do so,
run  
`pip install -r requirements.txt`

### Dataset Arrangement
The dataset folder need to be arranged in the following form before training:  
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
The `labels_npy` folder contains labels in the numpy array form. The shape
of the array should be two dimension: (width, height). Users can use matplotlib
and numpy methods to transform labels in image form to `.npy` form. This process
is not included in this code because of the diverse original file form of
labels.  
Note that the name of the label numpy file is the same as the name
of the image (`a.png`, `a.npy`). 

## Training
Before train a model, edit `config.py` with desired parameters.

The `config` is a multi-level dictionary. For example:

```
config = {
    'uhcs': {
        'default': gen_default('uhcs', n_class=4, size=(484, 645)),
        'v1': {'model': 'pixelnet'},
        'v2': {'model': 'unet', 'optimizer': 'SGD', 'lr': 1e-2}
    }
}
```
In this example, `uhcs` is the dataset name, `gen_default()` returns
the default parameters dictionary. The default parameters are:
- `root`: the dataset directory
- `n_class`: the number of classes
- `size`: the image size to resize into
- `batch_size`: the batch size during training (default: 1)
- `optimizer`: the type of the optimizer (default: Adam)
- `lr`: the initial learning rate (default: 1e-4)
- `epoch`: the total number of epochs (default: 60)  

The parameters defined in version (`v1, v2`) overwrite the corresponding
default value. `model` should be defined explicitly because there is no default model.
In this example, the parameters in version `v2` is the default parameters with modified
optimizer and learning rate.

To train a model, run `main.py` with the following argument:  
`python3 main.py dataset train version`

- `dataset`: the name of the dataset folder (eg: `uhcs`)
- `version`: the configuration version (eg: `v1`)

However, this setting does not save the model. To save the model during 
training, run  
`python3 main.py dataset train version --save`

## Evaluation
To evaluation a model on test images, run `main.py` with the following argument:  
`python3 main.py dataset evaluate version --test-folder TEST_FOLDER`
- dataset: the name of the dataset folder (eg: `uhcs`)
- version: the configuration version (eg: `v1`)
- TEST_FOLDER: the name of the test folder, default: `test`

For example, run  
`python3 main.py uhcs evaluate v1 --test-folder validate`
will evaluate the images under `validate` folder for `uhcs` dataset with
model trained in version `v1`.

## Related Paper
Segmentation of four microstructure constituents in ultrahigh carbon steel dataset: https://arxiv.org/abs/1805.08693
