import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from msdnet.dataloader import get_dataloaders
from msdnet.models.msdnet import MSDNet
from imta.models.msdnet_imta import IMTA_MSDNet
from model_params import (
    get_msdnet_default_args_cifar,
    get_imta_default_args_cifar,
    get_msdnet_default_args_imagenet,
    get_imta_default_args_imagenet,
)

import os
import random
from typing import Tuple, Dict, List
from collections import OrderedDict


def get_preds_per_exit(probs: torch.Tensor) -> Dict[int, torch.Tensor]:
    L = probs.shape[0]
    return {i: torch.argmax(probs, dim=2)[i, :] for i in range(L)}


def get_acc_per_exit(
    preds: Dict[int, torch.Tensor], targets: torch.Tensor
) -> List[float]:
    L = len(preds)
    return [(targets == preds[i]).sum() / len(targets) for i in range(L)]


def get_logits_targets(
    dataset: str,
    model: str,
    checkpoint: str,
    cuda: bool = True,
    logits_type: str = "test",
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dataset in ["cifar10", "cifar100", "imagenet"]
    assert model in ["msdnet", "imta"]

    if dataset.startswith("cifar"):
        assert logits_type in ["train", "val", "test"]
        if model == "msdnet":
            ARGS = get_msdnet_default_args_cifar(dataset)
        elif model == "imta":
            ARGS = get_imta_default_args_cifar(dataset)
    elif dataset == "imagenet":
        assert logits_type in ["train", "val"]
        if model == "msdnet":
            ARGS = get_msdnet_default_args_imagenet()
        elif model == "imta":
            ARGS = get_imta_default_args_imagenet()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # load pre-trained model
    device = torch.device("cuda" if cuda else "cpu")
    state = torch.load(f"{ARGS.save}/{checkpoint}", map_location=device)
    if model == "msdnet":
        model = MSDNet(args=ARGS)
        params = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
    elif model == "imta":
        model = IMTA_MSDNet(args=ARGS)
        params = OrderedDict()
        problematic_prefix = 'module.'
        for params_name, params_val in state['state_dict'].items():
            if params_name.startswith(problematic_prefix):
                params_name = params_name[len(problematic_prefix):]
            params[params_name] = params_val
        
    model.load_state_dict(params)
    model = model.to(device)
    model.eval()

    # data
    if (
        dataset == "imagenet" and logits_type == "val"
    ):  # imagenet val set (get_loaders function requires presence of train set)
        data_loader = get_imagenet_valid_loader(ARGS)
    else:
        dataloaders = get_dataloaders(ARGS)
        data_loader = dataloaders[["train", "val", "test"].index(logits_type)]

    # get logits and targets
    logits = []
    targets = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            output = [output] if not isinstance(output, list) else output

            logits.append(torch.stack(output))
            targets.append(y)

    logits = torch.cat(logits, dim=1).cpu()
    targets = torch.cat(targets).cpu()

    return logits, targets


def get_imagenet_valid_loader(args):
    valdir = os.path.join(args.data_root, "val")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    val_set = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    return val_loader


def raps_eenn(
    probs: np.ndarray,
    targets: np.ndarray,
    calib_size: float = 0.2,
    alpha: float = 0.05,
    lam_reg: float = 0.01,
    k_reg: float = 5,
    disallow_zero_sets: bool = False,
    rand: bool = True,
    seed: int = 0,
) -> Tuple[List, List]:
    """
    Code adapted from:
        https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/imagenet-raps.ipynb
    """
    L, N, C = probs.shape

    random.seed(seed)
    calib_ids = random.sample(range(N), int(calib_size * N))
    valid_ids = list(set(range(N)) - set(calib_ids))

    reg_vec = np.array(
        k_reg
        * [
            0,
        ]
        + (C - k_reg)
        * [
            lam_reg,
        ]
    )[None, :]

    sizes, coverages = [], []
    for exit in range(L):
        cal_smx = probs[exit, calib_ids, :]
        cal_labels = targets[calib_ids].cpu().numpy()
        n = len(cal_labels)

        val_smx = probs[exit, valid_ids, :]
        valid_labels = targets[valid_ids].cpu().numpy()
        n_valid = len(valid_labels)

        # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
        cal_pi = cal_smx.argsort(1)[:, ::-1]
        cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1)
        cal_srt_reg = cal_srt + reg_vec
        cal_L = np.where(cal_pi == cal_labels[:, None])[1]
        cal_scores = (
            cal_srt_reg.cumsum(axis=1)[np.arange(n), cal_L]
            - np.random.rand(n) * cal_srt_reg[np.arange(n), cal_L]
        )
        # Get the score quantile
        qhat = np.quantile(
            cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
        )
        # Deploy
        n_val = val_smx.shape[0]
        val_pi = val_smx.argsort(1)[:, ::-1]
        val_srt = np.take_along_axis(val_smx, val_pi, axis=1)
        val_srt_reg = val_srt + reg_vec
        indicators = (
            (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val, 1) * val_srt_reg)
            <= qhat
            if rand
            else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
        )
        if disallow_zero_sets:
            indicators[:, 0] = True
        conformal_sets = np.take_along_axis(indicators, val_pi.argsort(axis=1), axis=1)

        sizes.append(conformal_sets.sum(axis=1).mean())
        coverages.append(
            conformal_sets[np.arange(n_valid), valid_labels].sum() / n_valid
        )

    return sizes, coverages
