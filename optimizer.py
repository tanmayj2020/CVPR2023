# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from torch import optim as optim


def build_optimizer(model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    OPTIMIZER_MOMENTUM = 0.9
    BASE_LR = 6.25e-05
    WEIGHT_DECAY = 0.05
    OPTIMIZER_BETAS = (0.9, 0.999)
    OPTIMIZER_EPS = 1e-08
    opt_lower = 'AdamW'.lower() #config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=OPTIMIZER_MOMENTUM, nesterov=True,
                              lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=OPTIMIZER_EPS, betas=OPTIMIZER_BETAS,
                                lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
