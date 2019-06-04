# configurations


# generate default configuration given necessary infos and hyperparameters
def gen_default(dataset, n_class, size, batch_size=1, lr=1e-4,
                epoch=30, decay_epoch=(16,), print_freq=50):
    default = {
        'root': './data/' + dataset,
        'n_class': n_class,
        'size': size,
        'batch_size': batch_size,
        'lr': lr,
        'epoch': epoch,
        'decay_epoch': decay_epoch,
        'print_freq': print_freq
    }
    return default


config = {
    'uhcs': {
        'default': gen_default('uhcs', n_class=4, size=(484, 645)),
        'v1': {'model': 'pixelnet'}
    },
    'ECCI': {
        'default': gen_default('ECCI', n_class=2, size=(768, 1024)),
        'v1': {'model': 'pixelnet'}
    },
    'tomography': {
        'default': gen_default('tomography', n_class=2, size=(852, 852)),
        'v1': {'model': 'pixelnet'},
        'v2': {'model': 'unet'}
    }
}


# combine the default config with version config
def get_config(dataset, version):
    args = config[dataset]['default'].copy()
    args.update(config[dataset][version])
    args['name'] = dataset + '_' + version
    return args
