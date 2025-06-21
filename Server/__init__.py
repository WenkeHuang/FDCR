import importlib
import inspect
import os


def get_all_models():
    return [model.split('.')[0] for model in os.listdir('Server')
            if not model.find('__') > -1 and 'py' in model]


Server_NAME = {}
for model in get_all_models():
    mod = importlib.import_module('Server.' + model)
    class_name = [x for x in mod.__dir__() if 'type' in str(type(getattr(mod, x))) and 'ServerMethod' in str(
        inspect.getmro(getattr(mod, x))[1:])]
    for d in class_name:
        c = getattr(mod, d)
        Server_NAME[c.NAME] = c


def get_server_method(args, cfg):
    return Server_NAME[args.server](args, cfg)
