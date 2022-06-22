from argparse import ArgumentParser, Namespace
import yaml
import os
from os.path import splitext
import torch
import random
import numpy as np


class Arguments:
    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument("--dataset", default="uhcs")
        parser.add_argument("--config", default="default.yaml")
        parser.add_argument("--gpu_id", type=int, default=0)
        parser.add_argument("--seed", type=int, default=42)
        self.parser = parser

    def parse_args(self, verbose=False, use_random_seed=True):
        args = self.parser.parse_args()
        args.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

        # load default config and specific config to args for the dataset
        default_config_path = f"./configs/{args.dataset}/default.yaml"
        default_config = yaml.safe_load(open(f"{default_config_path}", 'r'))
        config_path = f"./configs/{args.dataset}/{args.config}"
        config = yaml.safe_load(open(f"{config_path}", 'r'))
        args = vars(args)
        args.update(default_config)
        args.update(config)
        args['split_info'] = Namespace(**args['split_info'])
        args['optimizer'] = Namespace(**args['optimizer'])
        if args['lr_scheduler']:
            args['lr_scheduler'] = Namespace(**args['lr_scheduler'])
        args = Namespace(**args)

        # compile basic information
        args.dataset_root = f"./data/{args.dataset}"
        assert os.path.exists(args.dataset_root), FileNotFoundError(args.dataset_root)
        args.img_dir = f"{args.dataset_root}/{args.img_folder}"
        args.label_dir = f"{args.dataset_root}/{args.label_folder}"

        args.experim_name = splitext(os.path.basename(args.config))[0]
        checkpoints_dir = f"./checkpoints/{args.dataset}/{args.experim_name}"
        self.update_checkpoints_dir(args, checkpoints_dir)

        # set seed
        if use_random_seed:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        if verbose:
            self.print_args(args)
        return args

    @staticmethod
    def update_checkpoints_dir(args, checkpoints_dir):
        args.checkpoints_dir = checkpoints_dir
        os.makedirs(args.checkpoints_dir, exist_ok=True)
        args.model_path = f"{args.checkpoints_dir}/model.pth"
        args.record_path = f"{args.checkpoints_dir}/train_record.csv"
        args.args_path = f"{args.checkpoints_dir}/args.yaml"
        args.val_result_path = f"{args.checkpoints_dir}/val_result.pkl"
        args.test_result_path = f"{args.checkpoints_dir}/test_result.pkl"
        args.pred_dir = f"{args.checkpoints_dir}/predictions"
        os.makedirs(args.pred_dir, exist_ok=True)

    @staticmethod
    def print_args(args):
        print(f"Configurations\n{'=' * 50}")
        [print(k, ':', v) for k, v in vars(args).items()]
        print('=' * 50)

    @staticmethod
    def save_args(args, path):
        with open(path, 'w') as file:
            yaml.dump(vars(args), file)


if __name__ == '__main__':
    arg_parser = Arguments()
    args = arg_parser.parse_args(verbose=True)
