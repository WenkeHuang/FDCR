import numpy as np
from Attack.backdoor.utils import backdoor_attack
from Attack.byzantine.utils import attack_dataset
from Server import get_server_method, Server_NAME
from Datasets.federated_dataset.single_domain import single_domain_dataset_name, get_single_domain_dataset
from Optims import Fed_Optim_NAMES, get_fed_method
from utils.conf import set_random_seed, config_path, log_path
from Datasets.federated_dataset.multi_domain import multi_domain_dataset_name, get_multi_domain_dataset
from Backbones import get_private_backbones
from utils.cfg import CFG as cfg, simplify_cfg, show_cfg
from utils.utils import ini_client_domain
from argparse import ArgumentParser
from utils.training import train
import datetime
import socket
import uuid
import setproctitle
import argparse
import os


def parse_args():
    parser = ArgumentParser(description='Federated Learning', allow_abbrev=False)
    parser.add_argument('--device_id', type=int, default=0, help='The Device Id for Experiment')
    '''
    Task: label_skew domain_skew
    '''
    parser.add_argument('--task', type=str, default='label_skew')
    '''
    label_skew:   fl_cifar10 fl_cifar100 fl_mnist fl_usps fl_fashionmnist fl_tinyimagenet
    '''
    parser.add_argument('--dataset', type=str, default='fl_cifar10',
                        help='Which scenario to perform experiments on.')
    '''
    Attack: byzantine backdoor None
    '''
    parser.add_argument('--attack_type', type=str, default='backdoor')

    '''
    Federated Method:  fedavg  fedprox  PrevAbsFedFish
    '''
    parser.add_argument('--optim', type=str, default='FedFish',
                        help='Federated Method name.', choices=Fed_Optim_NAMES)

    parser.add_argument('--rand_domain_select', type=bool, default=False, help='The Local Domain Selection')
    '''
    Aggregations Strategy Hyper-Parameter
    '''
    parser.add_argument('--server', type=str, default='OurRandomControl', choices=Server_NAME, help='The Option for averaging strategy')
    parser.add_argument('--seed', type=int, default=0, help='The random seed.')

    parser.add_argument('--csv_log', action='store_true', default=True, help='Enable csv logging')
    parser.add_argument('--result_path', default=log_path(), help='Enable csv logging')
    parser.add_argument('--csv_name', type=str, default=None, help='Predefine the csv name')
    parser.add_argument('--save_checkpoint', action='store_true', default=False)

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


def main(args=None):
    if args is None:
        args = parse_args()

    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    cfg_dataset_path = os.path.join(config_path(), 'Datasets', args.task, args.dataset, 'Default.yaml')
    cfg.merge_from_file(cfg_dataset_path)

    # cfg_method_path = os.path.join(config_path(), args.dataset, args.method + '.yaml')
    # if os.path.exists(cfg_method_path):
    #     cfg.merge_from_file(cfg_method_path)
    cfg.merge_from_list(args.opts)

    particial_cfg = simplify_cfg(args, cfg)

    show_cfg(args, particial_cfg)
    if args.seed is not None:
        set_random_seed(args.seed)

    '''
    Loading the dataset
    '''
    if args.dataset in multi_domain_dataset_name:
        private_dataset = get_multi_domain_dataset(args, particial_cfg)
    elif args.dataset in single_domain_dataset_name:
        private_dataset = get_single_domain_dataset(args, particial_cfg)

    if args.task == 'label_skew':
        private_dataset.get_data_loaders()
        client_domain_list = None

    elif args.task == 'domain_skew':
        client_domain_list = ini_client_domain(args.rand_domain_select, private_dataset.domain_list, particial_cfg.DATASET.parti_num)
        private_dataset.get_data_loaders(client_domain_list)

    if args.attack_type == 'byzantine':

        if args.dataset in multi_domain_dataset_name:
            particial_cfg.attack.dataset_type = 'multi_domain'

        elif args.dataset in single_domain_dataset_name:
            particial_cfg.attack.dataset_type = 'single_domain'

        bad_scale = int(particial_cfg.DATASET.parti_num * particial_cfg['attack'].bad_client_rate)
        good_scale = particial_cfg.DATASET.parti_num - bad_scale
        client_type = np.repeat(True, good_scale).tolist() + (np.repeat(False, bad_scale)).tolist()

        attack_dataset(args, particial_cfg, private_dataset, client_type)

    elif args.attack_type == 'backdoor':
        bad_scale = int(particial_cfg.DATASET.parti_num * particial_cfg['attack'].bad_client_rate)
        good_scale = particial_cfg.DATASET.parti_num - bad_scale
        client_type = np.repeat(True, good_scale).tolist() + (np.repeat(False, bad_scale)).tolist()

        backdoor_attack(args, particial_cfg, client_type, private_dataset, is_train=True)

        backdoor_attack(args, particial_cfg, client_type, private_dataset, is_train=False)

    '''
    Loading the Private Backbone
    '''
    priv_backbones = get_private_backbones(particial_cfg)

    '''
    Loading the Federated Optimizer
    '''
    fed_method = get_fed_method(priv_backbones, client_domain_list, args, particial_cfg)

    ''''
    Loading the Federated Aggregation
    '''
    fed_server = get_server_method(args, particial_cfg)
    if args.attack_type == 'byzantine':
        fed_method.client_type = client_type

    if args.csv_name is None:
        setproctitle.setproctitle('{}_{}_{}'.format(args.optim, args.server, args.dataset))
    else:
        setproctitle.setproctitle('{}_{}_{}_{}'.format(args.optim, args.server, args.dataset, args.csv_name))

    train(fed_method, fed_server, private_dataset, args, particial_cfg)


if __name__ == '__main__':
    main()
