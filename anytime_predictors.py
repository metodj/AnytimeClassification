import torch
from typing import Optional, Dict, List


def anytime_caching(logits: torch.Tensor) -> torch.Tensor:
    """
    probs: tensor of shape (L, N, C) where L is number of early exits,
        N is number of test samples, C is number of classes
    """
    L, N, _ = logits.shape

    probs = torch.softmax(logits, dim=2)

    max_probs = torch.zeros(N)
    max_ids = torch.zeros(N, dtype=torch.long)
    probs_caching = torch.zeros_like(probs)

    for exit in range(L):
        max_prob_current = probs[exit, :, :].max(dim=1).values
        update_mask = max_prob_current >= max_probs
        max_ids[update_mask] = exit
        max_probs[update_mask] = max_prob_current[update_mask]
        probs_caching[exit, :, :] = probs[max_ids, range(N), :]

    return probs_caching


def anytime_product(
    logits: torch.Tensor,
    thres_min: float = 0.0,
    weights: Optional[torch.Tensor] = None,
    fall_back: bool = False,
    thres_max: Optional[float] = None,
    softplus: bool = False,
) -> torch.Tensor:
    """
    logits: tensor of shape (L, N, C) where L is number of early exits,
        N is number of test samples, C is number of classes
    thres_min: threshold used in ReLU
    weights: (L,) weights for each early exit
    fall_back: if True, fall-back to a softmax predictive in case of
        a collapse to a zero distribution
    thres_max: threshold used for clipping logits from above
    softplus: if True, use softplus instead of (modified) ReLU
    """
    L, N, _ = logits.shape
    probs = logits.clone()

    if not softplus:
        if thres_max is not None:
            probs = torch.clamp(probs, max=thres_max)
        probs = torch.clamp(probs, min=thres_min)
    else:
        probs = torch.nn.functional.softplus(probs)

    if weights is not None:
        assert L == weights.shape[0], "Incompatible shapes between logits and weights"
        probs = probs.pow(weights.view(L, 1, 1))

    probs = torch.cumprod(probs, dim=0)

    if fall_back:
        sum_probs = probs.sum(dim=2, keepdim=True)
        zeros_mask = sum_probs.eq(0.0)
        probs = torch.where(zeros_mask, torch.softmax(logits, dim=2), probs / sum_probs)
    else:
        probs /= probs.sum(dim=2, keepdim=True)

    return probs


def conditional_monotonicity_check(
    y: torch.Tensor,
    probs: torch.Tensor,
    thresholds: List[float] = [0.01, 0.05, 0.1, 0.2, 0.5],
) -> Dict[float, float]:
    """ 
    y: classes of interest (N,), e.g., ground truth labels
    probs: tensor of shape (L, N, C) where L is number of early exits,
        N is number of test samples, C is number of classes
    thresholds: 
    """
    def diffs(arr: torch.tensor) -> torch.tensor:
        L = len(arr)
        diffs = []
        for i in range(L):
            for j in range(i + 1, L):
                diffs.append(arr[j] - arr[i])
        return torch.tensor(diffs)
    
    N = len(y)
    nr_no_decrease = {thres: 0 for thres in thresholds}
    for i in range(N):
        probs_i = probs[:, i, y[i]]
        diffs_i = diffs(probs_i)
        for thres in nr_no_decrease.keys():
            if torch.all(diffs_i >= -thres):
                nr_no_decrease[thres] += 1
    nr_decrease = {
        k: ((N - v) / N) * 100 for k, v in nr_no_decrease.items()
    }
    return nr_decrease



