import os
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models_depth.model import VPDDepth
import utils_depth.metrics as metrics
import utils_depth.logging as logging
from torchvision import datasets, transforms,models
from dataset.base_dataset import get_dataset
from configs.test_options import TestOptions
import utils
from ZoeDepth.zoedepth.models.builder import build_model
from ZoeDepth.zoedepth.utils.config import get_config
from ZoeDepth.zoedepth.utils.misc import colorize
# from ZoeDepth.evaluate import DepthDataLoader, evaluate
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']

from Depth2HHA.utils_hha.getCameraParam import getCameraParam
from Depth2HHA.getHHA import getHHA
def main():
    opt = TestOptions()
    args = opt.initialize().parse_args()
    print(args)

    utils.init_distributed_mode_simple(args)
    # print(args)
    device = torch.device(args.gpu)

    model = VPDDepth(args=args)

    # CPU-GPU agnostic settings

    cudnn.benchmark = True
    model.to(device)

    conf = get_config("zoedepth", "infer", dataset='nyu')
    model_zoe_n = build_model(conf)
    model_zoe_n.cuda()

    from collections import OrderedDict
    model_weight = torch.load(args.ckpt_dir)['model']
    if 'module' in next(iter(model_weight.items()))[0]:
        model_weight = OrderedDict((k[7:], v) for k, v in model_weight.items())
    model.load_state_dict(model_weight, strict=False)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    # Dataset setting
    dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path}
    dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

    data_dataset = get_dataset(**dataset_kwargs,is_train=False)

    # for numpy.save, no sampler
    data_loader = torch.utils.data.DataLoader(data_dataset, batch_size=1, sampler=None, pin_memory=False, num_workers=1, shuffle=False)


    validate(data_loader, model_zoe_n, model, device=device, args=args)
