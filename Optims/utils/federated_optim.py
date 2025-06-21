import copy
import numpy as np
import torch.nn as nn
import torch
from argparse import Namespace
from utils.conf import get_device
from utils.conf import checkpoint_path
from utils.utils import create_if_not_exists
import os


class FederatedOptim(nn.Module):
    """
    Federated learning model.
    """
    NAME = None
    def __init__(self, nets_list: list,client_domain_list: list,
                 args: Namespace,cfg) -> None:
        super(FederatedOptim, self).__init__()
        self.nets_list = nets_list
        self.args = args
        self.cfg = cfg
        self.client_domain_list = client_domain_list
        # For Online
        self.random_state = np.random.RandomState()
        self.online_num = np.ceil(self.cfg.DATASET.parti_num * self.cfg.DATASET.online_ratio).item()
        self.online_num = int(self.online_num)

        self.global_net = None
        self.device = get_device(device_id=self.args.device_id)

        self.local_epoch = self.cfg.OPTIMIZER.local_epoch
        self.local_lr = self.cfg.OPTIMIZER.local_train_lr
        self.weight_decay = self.cfg.OPTIMIZER.weight_decay

        self.train_loaders = None
        self.test_loaders = None
        self.net_cls_counts = None

        self.epoch_index = 0

        self.checkpoint_path = checkpoint_path() + self.args.dataset  + '/'
        create_if_not_exists(self.checkpoint_path)
        self.net_to_device()

        # self.fish_diff_dict = {}
        self.local_fish_dict = {}

    def net_to_device(self):
        for net in self.nets_list:
            net.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_scheduler(self):
        return

    def ini(self):
        pass


    def loc_update(self, priloader_list):
        pass

    def load_pretrained_nets(self):
        if self.load:
            for j in range(self.args.parti_num):
                pretrain_path = os.path.join(self.checkpoint_path, 'pretrain')
                save_path = os.path.join(pretrain_path, str(j) + '.ckpt')
                self.nets_list[j].load_state_dict(torch.load(save_path, self.device))
        else:
            pass

    def copy_nets2_prevnets(self):
        nets_list = self.nets_list
        prev_nets_list = self.prev_nets_list
        for net_id, net in enumerate(nets_list):
            prev_nets_list[net_id] = copy.deepcopy(net)

