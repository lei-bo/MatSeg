# configurations


# generate default configuration given necessary infos and hyperparameters
def gen_default(dataset, class_num, size, batch_size=1, lr=1e-4,
                epoch=30, decay_epoch=(16,), print_interval=50):
    default = {
        "root": "./data/" + dataset,
        "class_num": class_num,
        "size": size,
        "batch_size": batch_size,
        "lr": lr,
        "epoch": epoch,
        "decay_epoch": decay_epoch,
        "print_interval": print_interval
    }
    return default


config = {
    "uhcs": {
        "default": gen_default("uhcs", class_num=4, size=(484, 645)),
        "v1": {"model": "pixelnet", "save": False}
    },
    "ECCI": {
        "default": gen_default("ECCI", class_num=2, size=(768, 1024)),
        "v1": {"model": "pixelnet", "save": False}
    },
    "tomography": {
        "default": gen_default("tomography", class_num=2, size=(852, 852)),
        "v1": {"model": "pixelnet", "save": False}
    }
}


# combine the default config with version config
def get_config(dataset, version):
    args = config[dataset]["default"].copy()
    args.update(config[dataset][version])
    args["name"] = dataset + "_" + version
    return args
