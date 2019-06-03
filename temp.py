import os
import shutil
import numpy as np
from matplotlib import pyplot as plt


label_root = "./data/tomography/test/labels"
label_npy_dir = "./data/tomography/test/labels_npy"
shutil.rmtree(label_npy_dir)
os.mkdir(label_npy_dir)
label_names = os.listdir(label_root)
for name in label_names:
    label_dir = os.path.join(label_root, name)
    label = plt.imread(label_dir)
    label = label[:,:,0].astype(np.int8)
    print(label.shape, label.dtype, label)
    np.save(os.path.join(label_npy_dir, name[:-6]), label)