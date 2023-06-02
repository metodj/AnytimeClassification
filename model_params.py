import os
import argparse
from msdnet.args import arg_parser as msdnet_arg_parser
from imta.args import arg_parser as imta_arg_parser


def parse_args(arg_parser):
    args = arg_parser.parse_args(args=[])

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.grFactor = list(map(int, args.grFactor.split("-")))
    args.bnFactor = list(map(int, args.bnFactor.split("-")))
    args.nScales = len(args.grFactor)

    if args.use_valid:
        args.splits = ["train", "val", "test"]
    else:
        args.splits = ["train", "val"]

    if args.data == "cifar10":
        args.num_classes = 10
    elif args.data == "cifar100":
        args.num_classes = 100
    else:
        args.num_classes = 1000

    return args


def get_msdnet_default_args_cifar(dataset: str) -> argparse.Namespace:
    assert dataset in ["cifar10", "cifar100"]
    ARGS = parse_args(msdnet_arg_parser)
    ARGS.data_root = "data"
    ARGS.save = f"pretrained_models/msdnet/{dataset}"
    ARGS.data = dataset
    ARGS.arch = "msdnet"
    ARGS.batch_size = 64
    ARGS.epochs = 300
    ARGS.nBlocks = 7
    ARGS.stepmode = "even"
    ARGS.base = 4
    ARGS.nChannels = 16
    ARGS.j = 16
    ARGS.num_classes = 100 if dataset == "cifar100" else 10
    ARGS.step = 2
    ARGS.use_valid = True
    ARGS.splits = ["train", "val", "test"]
    return ARGS


def get_msdnet_default_args_imagenet(step=4) -> argparse.Namespace:
    assert step in [4, 7]
    ARGS = parse_args(msdnet_arg_parser)
    ARGS.data_root = "/data/ImageNet"
    ARGS.data = "ImageNet"
    ARGS.save = "pretrained_models/msdnet/imagenet"
    ARGS.arch = "msdnet"
    ARGS.batch_size = 64
    ARGS.epochs = 90
    ARGS.nBlocks = 5
    ARGS.stepmode = "even"
    ARGS.base = step
    ARGS.nChannels = 32
    ARGS.growthRate = 16
    ARGS.bnFactor = [1, 2, 4, 4]
    ARGS.grFactor = [1, 2, 4, 4]
    ARGS.j = 16
    ARGS.num_classes = 1000
    ARGS.step = step
    ARGS.use_valid = True
    # ARGS.splits = ["train", "val", "test"]
    ARGS.splits = ["val"]
    ARGS.nScales = len(ARGS.grFactor)
    return ARGS


def get_imta_default_args_cifar(
    dataset: str, GE_model: str = "checkpoint_299.pth.tar"
) -> argparse.Namespace:
    assert dataset in ["cifar10", "cifar100"]
    ARGS = parse_args(imta_arg_parser)
    ARGS.data_root = "data"
    ARGS.data = dataset
    ARGS.save = f"pretrained_models/imta/{dataset}/model"
    ARGS.arch = "IMTA_MSDNet"
    ARGS.grFactor = [1, 2, 4]
    ARGS.bnFactor = [1, 2, 4]
    ARGS.growthRate = 6
    ARGS.batch_size = 64
    ARGS.epochs = 300
    ARGS.nBlocks = 7
    ARGS.stepmode = "even"
    ARGS.base = 4
    ARGS.nChannels = 16
    ARGS.num_classes = 10 if ARGS.data == "cifar10" else 100
    ARGS.step = 2
    ARGS.use_valid = True
    ARGS.splits = ["train", "val", "test"]
    ARGS.nScales = len(ARGS.grFactor)
    ARGS.T = 1.0
    ARGS.gamma = 0.1
    ARGS.pretrained = f"pretrained_models/imta/{dataset}/model_GE/{GE_model}"
    return ARGS


def get_imta_default_args_imagenet(
    GE_model: str = "checkpoint_089.pth.tar",
) -> argparse.Namespace:
    ARGS = parse_args(imta_arg_parser)
    ARGS.data_root = "/data/ImageNet"
    ARGS.data = "ImageNet"
    ARGS.save = "pretrained_models/imta/imagenet/model"
    ARGS.arch = "IMTA_MSDNet"
    ARGS.grFactor = [1, 2, 4, 4]
    ARGS.bnFactor = [1, 2, 4, 4]
    ARGS.growthRate = 16
    ARGS.batch_size = 350
    ARGS.epochs = 90
    ARGS.nBlocks = 5
    ARGS.stepmode = "even"
    ARGS.base = 4
    ARGS.nChannels = 32
    ARGS.num_classes = 1000
    ARGS.step = 4
    ARGS.use_valid = True
    # ARGS.splits = ['train', 'val', 'test']
    ARGS.splits = ["val"]
    ARGS.nScales = len(ARGS.grFactor)
    ARGS.T = 1.0
    ARGS.gamma = 0.1
    ARGS.pretrained = f"pretrained_models/imta/imagenet/model_GE/{GE_model}"
    return ARGS
