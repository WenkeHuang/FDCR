import importlib
import inspect
import os

def get_all_models():
    return [model.split('.')[0] for model in os.listdir('Optims')
            if not model.find('__') > -1 and 'py' in model]

Fed_Optim_NAMES = {}
for model in get_all_models():
    mod = importlib.import_module('Optims.' + model)
    class_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'FederatedOptim' in str(
        inspect.getmro(getattr(mod, x))[1:])]
    for d in class_name:
        c = getattr(mod, d)
        Fed_Optim_NAMES[c.NAME] = c

def get_fed_method(nets_list, client_domain_list, args, cfg):
    return Fed_Optim_NAMES[args.optim](nets_list, client_domain_list, args, cfg)
