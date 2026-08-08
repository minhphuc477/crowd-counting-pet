from .backbones import *
from .transformer import *
from .pet import build_pet
from .vgg_ebc_point import build_vgg_ebc_point

def build_model(args):
    if getattr(args, 'model_family', 'pet') == 'vgg_ebc_point':
        return build_vgg_ebc_point(args)
    return build_pet(args)

