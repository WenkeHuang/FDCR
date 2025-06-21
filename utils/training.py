import copy
from Attack.byzantine.utils import attack_net_para
from Optims.utils.federated_optim import FederatedOptim
from utils.logger import CsvWriter
from torch.utils.data import DataLoader
import torch
import numpy as np
from utils.utils import log_msg
from typing import Tuple


def cal_top_one_five(net, test_dl, device,method_name):
    net.eval()
    correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
    for batch_idx, (images, labels) in enumerate(test_dl):
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)
            if method_name =='FedETF':
                norm_features = net.norm_features(images)
                outputs = torch.matmul(norm_features, net.proto_classifier.proto.to(device))
            else:
                outputs = net(images)
            _, max5 = torch.topk(outputs, 5, dim=-1)
            labels = labels.view(-1, 1)
            top1 += (labels == max5[:, 0:1]).sum().item()
            top5 += (labels == max5).sum().item()
            total += labels.size(0)
    net.train()
    top1acc = round(100 * top1 / total, 2)
    top5acc = round(100 * top5 / total, 2)
    return top1acc, top5acc


def global_in_evaluation(model: FederatedOptim, test_loader: dict, in_domain_list: list):
    in_domain_accs = []
    for in_domain in in_domain_list:
        global_net = model.global_net
        global_net.eval()

        test_domain_dl = test_loader[in_domain]
        top1acc, _ = cal_top_one_five(net=global_net, test_dl=test_domain_dl, device=model.device)
        in_domain_accs.append(top1acc)
        global_net.train()
    mean_in_domain_acc = round(np.mean(in_domain_accs, axis=0), 3)
    return in_domain_accs, mean_in_domain_acc


def fill_blank(net_cls_counts, classes):
    class1 = [i for i in range(classes)]

    for client, dict_i in net_cls_counts.items():
        if len(dict_i.keys()) == 10:
            continue
        else:
            for i in class1:
                if i not in dict_i.keys():
                    dict_i[i] = 0

    return net_cls_counts


def train(fed_method, fed_server, private_dataset, args, cfg) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, cfg)

    if hasattr(fed_method, 'ini'):
        fed_method.ini()
        fed_server.ini()

    if args.task == 'label_skew':
        mean_in_domain_acc_list = []
        if args.attack_type == 'None':
            contribution_match_degree_list = []
        fed_method.net_cls_counts = fill_blank(private_dataset.net_cls_counts, cfg.DATASET.n_classes)
    elif args.task == 'domain_skew':
        in_domain_accs_dict = {}  # Query-Client Accuracy \bm{\mathcal{A}}}^{u}
        mean_in_domain_acc_list = []  # Cross-Client Accuracy A^U \bm{\mathcal{A}}}^{\mathcal{U}
        performance_variane_list = []
        if args.attack_type == 'None':
            contribution_match_degree_list = []
    if args.attack_type == 'backdoor':
        attack_success_rate = []

    communication_epoch = cfg.DATASET.communication_epoch

    for epoch_index in range(communication_epoch):
        fed_method.epoch_index = epoch_index
        fed_server.epoch_index = epoch_index

        if hasattr(fed_method, 'loc_update'):
            fed_method.val_loader = private_dataset.val_loader
            fed_method.loc_update(private_dataset.train_loaders)

        if args.attack_type == 'byzantine':
            attack_net_para(args, cfg, fed_method)

        fed_server.server_update(online_clients_list=fed_method.online_clients_list,
                                 priloader_list=private_dataset.train_loaders,
                                 client_domain_list=fed_method.client_domain_list, global_net=fed_method.global_net,
                                 nets_list=fed_method.nets_list, val_loader=private_dataset.val_loader,
                                 epoch_index=epoch_index, local_fish_dict=fed_method.local_fish_dict)

        if args.csv_log:
            if hasattr(fed_server, 'div_score'):
                csv_writer.write_metric(fed_server.div_score.cpu().numpy(), epoch_index,'div_score')
            if hasattr(fed_server, 'aggregation_weight'):
                csv_writer.write_metric(fed_server.aggregation_weight, epoch_index,'aggregation_weight')

        # Server
        if 'mean_in_domain_acc_list' in locals() and args.task == 'label_skew':
            print("eval mean_in_domain_acc_list")
            top1acc, _ = cal_top_one_five(fed_method.global_net, private_dataset.test_loader, fed_method.device,fed_method.NAME)
            mean_in_domain_acc_list.append(top1acc)
            if args.csv_name == None:
                print(log_msg(f'The {epoch_index} Epoch: Acc:{top1acc} Optim:{args.optim} Server:{args.server} Dataset:{args.dataset} Beta:{cfg.DATASET.beta}', "TEST"))
            else:
                print(log_msg(f'The {epoch_index} Epoch: Acc:{top1acc} Optim:{args.optim} Server:{args.server} CSV:{args.csv_name} Dataset:{args.dataset} Beta:{cfg.DATASET.beta}', "TEST"))
        if 'attack_success_rate' in locals():
            top1acc, _ = cal_top_one_five(fed_method.global_net, private_dataset.backdoor_test_loader, fed_method.device,fed_method.NAME)
            attack_success_rate.append(top1acc)
            if args.csv_name == None:
                print(log_msg(f'The {epoch_index} Epoch: attack success rate:{top1acc} Optim:{args.optim} Server:{args.server} Dataset:{args.dataset} Beta:{cfg.DATASET.beta}', "ROBUST"))
            else:
                print(log_msg(f'The {epoch_index} Epoch: attack success rate:{top1acc} Optim:{args.optim} Server:{args.server} CSV:{args.csv_name} Dataset:{args.dataset} Beta:{cfg.DATASET.beta}', "ROBUST"))
        if args.csv_log:
            if args.save_checkpoint:
                torch.save(fed_method.global_net.state_dict(), csv_writer.para_path + '/model.pth')
                print('SAVE!')
    if args.csv_log:
        if args.task == 'label_skew':
            csv_writer.write_acc(mean_in_domain_acc_list, name='in_domain', mode='MEAN')
            if args.attack_type == 'None':
                csv_writer.write_acc(contribution_match_degree_list, name='contribution_fairness', mode='MEAN')

        if args.attack_type == 'backdoor':
            csv_writer.write_acc(attack_success_rate, name='attack_success_rate', mode='MEAN')

        if args.save_checkpoint:
            torch.save(fed_method.global_net.state_dict(), csv_writer.para_path + '/model_final.pth')
