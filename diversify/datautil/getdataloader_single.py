# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# coding=utf-8
import numpy as np
import uci_loader
from torch.utils.data import DataLoader

import datautil.actdata.util as actutil
from datautil.util import combindataset, subdataset

import datautil.actdata.cross_people as cross_people

task_act = {'cross_people': cross_people}


def get_dataloader(args, tr, val, tar):
    train_loader = DataLoader(dataset=tr, batch_size=args.batch_size,
                              num_workers=args.N_WORKERS, drop_last=False, shuffle=True)
    train_loader_noshuffle = DataLoader(
        dataset=tr, batch_size=args.batch_size, num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    valid_loader = DataLoader(dataset=val, batch_size=args.batch_size,
                              num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    target_loader = DataLoader(dataset=tar, batch_size=args.batch_size,
                               num_workers=args.N_WORKERS, drop_last=False, shuffle=False)
    return train_loader, train_loader_noshuffle, valid_loader, target_loader


def get_act_dataloader(args):
    source_datasetlist = []
    target_datalist = []
    pcross_act = task_act[args.task]

    tmpp = args.act_people[args.dataset]
    args.domain_num = len(tmpp)
    for i, item in enumerate(tmpp):
        tdata = pcross_act.ActList(
            args, args.dataset, args.data_dir, item, i, transform=actutil.act_train())
        if i in args.test_envs:
            target_datalist.append(tdata)
        else:
            source_datasetlist.append(tdata)
            if len(tdata)/args.batch_size < args.steps_per_epoch:
                args.steps_per_epoch = len(tdata)/args.batch_size
            if args.dataset == 'uci_har':
                return uci_loader.get_uci_har_dataloader(args)

    # Hold out one source environment for validation
    val_env_idx = 0  # Or choose dynamically, e.g., args.val_env
    source_envs_train = [env for i, env in enumerate(source_datasetlist) if i != val_env_idx]
    val_env = source_datasetlist[val_env_idx]
    
    tr = combindataset(args, source_envs_train)  # Training = other source environments
    val = val_env  # Validation = held-out source environment
    targetdata = combindataset(args, target_datalist)  # Target = test environments
    
    # Update steps_per_epoch (optional, adjust if needed)
    args.steps_per_epoch = len(tr) // args.batch_size
    train_loader, train_loader_noshuffle, valid_loader, target_loader = get_dataloader(
        args, tr, val, targetdata)
    return train_loader, train_loader_noshuffle, valid_loader, target_loader, tr, val, targetdata
