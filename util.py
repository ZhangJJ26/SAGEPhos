import os
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch import distributed as dist
from torch.optim import lr_scheduler

from torchdrug import core, utils, datasets, models, tasks
from torchdrug.utils import comm
from torch.optim.swa_utils import AveragedModel


logger = logging.getLogger(__file__)


def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.task["class"], cfg.dataset["class"], cfg.task.model["class"],
                               time.strftime("%Y-%m-%d-%H-%M-%S"))
    # example:/root/scratch/aaai24_output/MultipleBinaryClassification/EnzymeCommission/FusionNetwork/2024-02-01-02-34-57

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    ast = env.parse(raw) # ast: abstract syntax tree，是一种树状的数据结构，用来表示代码的语法结构
    vars = meta.find_undeclared_variables(ast) # 返回一个set，包含ast中所有未声明的变量
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-c", "--config", help="yaml configuration file", required=True, default="config/EC/esm_gearnet.yaml")
    parser.add_argument("-c", "--config", help="yaml configuration file", default="/chunbin2/phos_zhijiang/ESM-GearNet/config/phos/esm_gearnet.yaml")
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, default="null")
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def build_downstream_solver(cfg, dataset):
    # 如果dataset是一个tuple，那么就是train_set, valid_set, test_set，否则就是dataset.split()
    if isinstance(dataset, tuple):
        train_set, valid_set, test_set = dataset
    else:
        train_set, valid_set, test_set = dataset.split()
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))
    print(test_set)

    if cfg.task["class"] == 'MultipleBinaryClassification':
        cfg.task.task = [_ for _ in range(len(dataset.tasks))]
    else:
        cfg.task.task = dataset.tasks
    task = core.Configurable.load_config_dict(cfg.task)

    cfg.optimizer.params = task.parameters()        
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)

    if "scheduler" not in cfg:
        scheduler = None
    elif cfg.scheduler["class"] == "ReduceLROnPlateau":
        cfg.scheduler.pop("class")
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        cfg.engine.scheduler = scheduler

    # solver = core.Engine_3(task, train_set, valid_set, test_set, optimizer, **cfg.engine)
    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)

    if "lr_ratio" in cfg:
        cfg.optimizer.params = [
            {'params': solver.model.model.parameters(), 'lr': cfg.optimizer.lr * cfg.lr_ratio},
            {'params': solver.model.mlp.parameters(), 'lr': cfg.optimizer.lr}
        ]
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        solver.optimizer = optimizer
    elif "sequence_model_lr_ratio" in cfg:
        # assert cfg.task.model["class"] == "FusionNetwork" or cfg.task.model["class"] == "FusionNetwork_esm" or cfg.task.model["class"] == "FusionNetwork_stru"
        cfg.optimizer.params = [
            {'params': solver.model.model.sequence_model.parameters(), 'lr': cfg.optimizer.lr * cfg.sequence_model_lr_ratio},
            {'params': solver.model.model.structure_model.parameters(), 'lr': cfg.optimizer.lr * cfg.structure_model_lr_ratio},
            {'params': solver.model.mlp_1.parameters(), 'lr': cfg.optimizer.lr},
            {'params': solver.model.model.gated_feature_composer.parameters(), 'lr': cfg.optimizer.lr},
            {'params': solver.model.model.res_info_composer.parameters(), 'lr': cfg.optimizer.lr},
            {'params': [solver.model.model.a], 'lr': cfg.optimizer.lr},
            {'params': solver.model.model.phos_site_emb.parameters(), 'lr': cfg.optimizer.lr},
            {'params': solver.model.model.batch_norm_esm.parameters(), 'lr': cfg.optimizer.lr},
            {'params': solver.model.model.batch_norm_stru.parameters(), 'lr': cfg.optimizer.lr * cfg.structure_model_lr_ratio},
            {'params': solver.model.model._gated_feature_composer.parameters(), 'lr': cfg.optimizer.lr},
            {'params': solver.model.model._res_info_composer.parameters(), 'lr': cfg.optimizer.lr},
            {'params': [solver.model.model._a], 'lr': cfg.optimizer.lr},
            {'params': solver.model.model._phos_site_emb.parameters(), 'lr': cfg.optimizer.lr},
            # {'params': solver.model.model.attention.parameters(), 'lr': cfg.optimizer.lr},
            # {'params': [solver.model.model.position_encoding], 'lr': cfg.optimizer.lr},
            # {'params': [solver.model.model.a_k], 'lr': cfg.optimizer.lr},
            # {'params': solver.model.model.gated_feature_composer_k.parameters(), 'lr': cfg.optimizer.lr},
            # {'params': solver.model.model.res_info_composer_k.parameters(), 'lr': cfg.optimizer.lr},
            # {'params': solver.model.model.attn.parameters(), 'lr': cfg.optimizer.lr},
            # {'params': solver.model.model.node_linear.parameters(), 'lr': cfg.optimizer.lr},
            # {'params': [solver.model.model.seq_weight], 'lr': cfg.optimizer.lr},
            # {'params': [p for p in solver.model.loss_weight if p not in solver.model.sequence_model.parameters() and p not in solver.model.model.structure_model.parameters() 
            #             and p not in solver.model.mlp_1.parameters()] and p not in solver.model.model.gated_feature_composer.parameters() and p not in solver.model.model.res_info_composer.parameters()
            #             and p not in [solver.model.model.a] and p not in solver.model.model.phos_site_emb.parameters(), 'lr': cfg.optimizer.lr},
        ]
        # {'params': solver.model.swiglu.parameters(), 'lr': cfg.optimizer.lr},
        # freeze sequence_model and structure_model
        for p in solver.model.model.sequence_model.parameters():
            p.requires_grad = False
        # ------------------------------
        for p in solver.model.model.structure_model.parameters():
            p.requires_grad = False
        # ------------------------------
        # for p in solver.model.model.structure_model.parameters():
        #     p.requires_grad = False
        print("可学习参数量：", sum(p.numel() for p in solver.model.model.parameters() if p.requires_grad))
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        solver.optimizer = optimizer
        

    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    elif scheduler is not None:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        solver.scheduler = scheduler

    if cfg.get("checkpoint") is not None:
        solver.load(cfg.checkpoint)

    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
        # cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        # model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
        # task.model.load_state_dict(model_dict, strict=True)
        # solver.load(cfg.model_checkpoint, load_optimizer = False)
        solver.load(cfg.model_checkpoint)
    
    return solver, scheduler


def build_pretrain_solver(cfg, dataset):
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning("#dataset: %d" % (len(dataset)))

    task = core.Configurable.load_config_dict(cfg.task)
    if "fix_sequence_model" in cfg:
        if cfg.task["class"] == "Unsupervised":
            model_dict = cfg.task.model.model
            model = task.model.model
        else:
            model_dict = cfg.task.model 
            model = task.model
        assert model_dict["class"] == "FusionNetwork" or model_dict["class"] == "FusionNetwork_esm" or model_dict["class"] == "FusionNetwork_stru"
        for p in model.sequence_model.parameters():
            p.requires_grad = False
    cfg.optimizer.params = [p for p in task.parameters() if p.requires_grad]
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    solver = core.Engine(task, dataset, None, None, optimizer, **cfg.engine)
    
    return solver
