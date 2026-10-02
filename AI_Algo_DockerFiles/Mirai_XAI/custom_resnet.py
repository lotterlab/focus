# Shri Krishna: Sharanam Mam Jai Ambe Jai Amumaiya Jai Devudada Shri Ganeshay Nama: Shri Saraswatiyay Nama:
from torch import nn

from onconet.models.factory import RegisterModel, load_pretrained_weights, get_layers
from onconet.models.default_resnets import load_pretrained_model
from onconet.models.resnet_base import ResNet

@RegisterModel("custom_resnet")
class CustomResnet(nn.Module):
    def __init__(self, args):
        super(CustomResnet, self).__init__()
        layers = get_layers(args.block_layout)
        self._model = ResNet(layers, args)
        model_name = args.pretrained_imagenet_model_name
        if args.pretrained_on_imagenet:
            load_pretrained_weights(self._model,
                                    load_pretrained_model(model_name))

    def forward(self, x, risk_factors=None, batch=None, 
                CC_index=4, y_cc_l=4, y_cc_u=4, x_cc_l=4, x_cc_u=4,
                MLO_index=4, y_mlo_l=4, y_mlo_u=4, x_mlo_l=4, x_mlo_u=4):
        return self._model(x, risk_factors=risk_factors, batch=None, 
                           CC_index=CC_index, y_cc_l=y_cc_l, 
                           y_cc_u=y_cc_u, x_cc_l=x_cc_l, x_cc_u=x_cc_u,
                           MLO_index=MLO_index, y_mlo_l=y_mlo_l, y_mlo_u=y_mlo_u, 
                           x_mlo_l=x_mlo_l, x_mlo_u=x_mlo_u)

    def cuda(self, device=None):
        self._model = self._model.cuda(device)
        return self