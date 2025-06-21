from Server.utils.server_methods import ServerMethod


class Equal(ServerMethod):
    NAME = 'Equal'

    def __init__(self, args, cfg):
        super(Equal, self).__init__(args, cfg)

    def weight_calculate(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        freq =  [1/len(online_clients_list) for _ in range(len(online_clients_list))]
        return freq

    def server_update(self, **kwargs):
        online_clients_list = kwargs['online_clients_list']
        priloader_list = kwargs['priloader_list']
        global_net = kwargs['global_net']
        nets_list = kwargs['nets_list']

        freq = self.weight_calculate(online_clients_list=online_clients_list, priloader_list=priloader_list)

        self.agg_parts(online_clients_list=online_clients_list, nets_list=nets_list,
                                  global_net=global_net, freq=freq, except_part=[], global_only=False)
        return freq
