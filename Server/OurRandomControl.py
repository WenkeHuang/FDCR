from Server.utils.server_methods import ServerMethod
from utils.utils import row_into_parameters
from torch import optim, nn
from tqdm import tqdm
import numpy as np
import torch
import copy
import torch.nn.functional as F
from utils.finch import FINCH


class OurRandomControl(ServerMethod):
    NAME = 'OurRandomControl'

    def __init__(self, args, cfg):
        super(OurRandomControl, self).__init__(args, cfg)
        self.learning_rate = self.cfg.OPTIMIZER.local_train_lr
        self.div_score = None
        self.aggregation_weight = None

    def server_update(self, **kwargs):

        online_clients_list = kwargs['online_clients_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']

        default_net = copy.deepcopy(global_net) # 获取默认聚合方向

        priloader_list = kwargs['priloader_list']

        freq = self.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)
        # 获取 假定的更新方向
        self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                       global_net=default_net, freq=freq, except_part=[], global_only=True)

        bad_scale = int(self.cfg.DATASET.parti_num * self.cfg['attack'].bad_client_rate)
        good_scale = self.cfg.DATASET.parti_num - bad_scale
        client_type = np.repeat(True, good_scale).tolist() + (np.repeat(False, bad_scale)).tolist()

        local_fish_dict = kwargs['local_fish_dict']
        prev_net = copy.deepcopy(global_net)
        vectorize_nets_list = []
        for query_net in nets_list:
            vectorize_net = torch.cat([p.view(-1) for p in query_net.parameters()]).detach()
            vectorize_nets_list.append(vectorize_net)
        # 参数化上一轮的网络
        prev_vectorize_net = torch.cat([p.view(-1) for p in prev_net.parameters()]).detach()

        grad_list = []
        weight_list = []
        fish_list = []
        for query_index, _ in enumerate(nets_list):
            grad_list.append((prev_vectorize_net - vectorize_nets_list[query_index]) / self.learning_rate)
            query_fish_dict = local_fish_dict[query_index]
            query_fish = torch.cat([p.view(-1) for p in query_fish_dict.values()]).detach()
            if not client_type[query_index]:
                query_fish = torch.randn_like(query_fish)
            fish_list.append(query_fish)
            norm_fish = (query_fish - torch.min(query_fish)) / (torch.max(query_fish) - torch.min(query_fish))
            weight_list.append(norm_fish)

        weight_grad_list = []
        for query_index, _ in enumerate(nets_list):
            query_grad = grad_list[query_index]
            query_weight = weight_list[query_index]
            weight_grad_list.append(torch.mul(query_grad, query_weight))

        assert len(weight_grad_list) == len(freq), "张量列表和权重列表的长度必须相同"

        weight_global_grad = torch.zeros_like(weight_grad_list[0])
        # 客户端规模 x 特征长度
        for weight_client_grad, client_freq in zip(weight_grad_list, freq):
            weight_global_grad += weight_client_grad * client_freq

        div_score = []
        for query_index, _ in enumerate(nets_list):
            div_score.append(
                F.pairwise_distance(weight_grad_list[query_index].view(1, -1), weight_global_grad.view(1, -1), p=2))

        div_score = torch.tensor(div_score).view(-1, 1)
        fin = FINCH()
        fin.fit(div_score)

        if len(fin.partitions) == 0:
            reconstructed_freq = freq
        else:
            select_partitions = (fin.partitions)['parition_0']
            evils_center = max(select_partitions['cluster_centers'])
            evils_center_idx = np.where(select_partitions['cluster_centers'] == evils_center)[0]
            evils_idx = select_partitions['cluster_core_indices'][int(evils_center_idx)]
            benign_idx = [i for i in range(len(online_clients_list)) if i not in evils_idx]

            print('benign',benign_idx,'evil',evils_idx)
            freq[evils_idx] = 0
            reconstructed_freq = freq / sum(freq)

            for i in benign_idx:
                curr_net = nets_list[i]
                norm_weight = weight_list[i]
                index = 0
                for name, curr_param in curr_net.state_dict().items():
                    prev_para = prev_net.state_dict()[name].detach()
                    delta = (prev_para - curr_param.detach())
                    param_number = prev_para.numel()
                    param_size = prev_para.size()

                    weight_para = norm_weight[index:index + param_number].reshape(param_size).to(self.device)
                    weight_para = torch.nn.functional.sigmoid(weight_para) * 2

                    weight_delta = torch.mul(delta, weight_para)
                    index += param_number
                    curr_param.data.copy_(prev_para - weight_delta)
                nets_list[i] = curr_net

        self.div_score = (div_score)
        self.aggregation_weight = (reconstructed_freq)
        self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                       global_net=global_net, freq=reconstructed_freq, except_part=[], global_only=False)
        return freq
