# configurations
import sys


# generate default configuration given necessary infos and hyperparameters
def gen_default(dataset, n_class, size, batch_size=1, lr=1e-4,
                epoch=60):
    default = {
        'root': './data/' + dataset,
        'n_class': n_class,
        'size': size,
        'batch_size': batch_size,
        'optimizer': 'Adam',
        'lr': lr,
        'epoch': epoch,
    }
    return default


config = {
    'uhcs': {
        'default': gen_default('uhcs', n_class=4, size=(484, 645)),
        'v1': {'model': 'pixelnet'},
        'v2': {'model': 'unet'},
        'v3': {'model': 'segnet'}
    },
    'tomography': {
        'default': gen_default('tomography', n_class=2, size=(852, 852)),
        'v1': {'model': 'pixelnet',  'batch_size': 1},
        'v2': {'model': 'unet'},
        'v3': {'model': 'segnet'}
    }
}


# combine the default config with version config
def get_config(dataset, version):
    try:
        args = config[dataset]['default'].copy()
    except KeyError:
        print('dataset %s does not exist' % dataset)
        sys.exit(1)
    try:
        args.update(config[dataset][version])
    except KeyError:
        print('version %s is not defined' % version)
    args['name'] = dataset + '_' + version
    return args
