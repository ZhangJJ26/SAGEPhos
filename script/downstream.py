import os
import sys
import math
import pprint
import random

import numpy as np

import torch
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils import data as torch_data

from torchdrug import core, models, tasks, datasets, utils, data
from torchdrug.utils import comm, pretty

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util
from gearnet import model, dataset, task
import wandb


def train_and_validate(cfg, solver, scheduler):
    if cfg.train.num_epoch == 0:
        return

    step = math.ceil(cfg.train.num_epoch / 100)
    best_result = float("-inf")
    best_epoch = -1
    
    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.train(**kwargs)
        wandb.log({"lr": solver.optimizer.param_groups[0]["lr"]})

        metric = solver.evaluate("valid")
        wandb.log(metric)

        result = metric[cfg.metric]
        if result > best_result:
            best_result = result
            best_epoch = solver.epoch
            solver.save("model_epoch_%d.pth" % solver.epoch)
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(result)

    solver.load("model_epoch_%d.pth" % best_epoch)
    return solver


def test(cfg, solver):
    return solver.evaluate("test")


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    seed = args.seed
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver, scheduler = util.build_downstream_solver(cfg, dataset)

    if cfg.model_checkpoint is not None:
        print("Loading model from checkpoint:", cfg.model_checkpoint)
        test(cfg, solver)
    else:
        print("Training")
        train_and_validate(cfg, solver, scheduler)
        test(cfg, solver)
