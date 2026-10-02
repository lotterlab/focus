# Shri Krishna: Sharanam Mam Jai Ambe Jai Amumaiya Jai Devudada Shri Ganeshay Nama: Shri Saraswatiyay Nama:
from onconet.models.factory import load_model, RegisterModel, get_model_by_name
import math
import torch
import torch.nn as nn
import pdb
import numpy as np

@RegisterModel("mirai_full")
class MiraiFull(nn.Module):

    def __init__(self, args):
        super(MiraiFull, self).__init__()
        self.args = args
        if args.img_encoder_snapshot is not None:
            self.image_encoder = load_model(args.img_encoder_snapshot, args, do_wrap_model=False)
        else:
            self.image_encoder = get_model_by_name('custom_resnet', False, args)

        if hasattr(self.args, "freeze_image_encoder") and self.args.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.image_repr_dim = self.image_encoder._model.args.img_only_dim
        if args.transformer_snapshot is not None:
            self.transformer = load_model(args.transformer_snapshot, args, do_wrap_model=False)
        else:
            args.precomputed_hidden_dim = self.image_repr_dim
            self.transformer = get_model_by_name('transformer', False, args)
        args.img_only_dim = self.transformer.args.transfomer_hidden_dim

    # def forward(self, x, risk_factors=None, batch=None):
    def forward(self, x, risk_factors=None, batch=None, 
                img_index=4, y_l=4, y_u=4, x_l=4, x_u=4):
        print("Input:", x.size())

        img_index=self.args.img_index
        y_l=self.args.y_l
        y_u=self.args.y_u
        x_l=self.args.x_l
        x_u=self.args.x_u
        
        B, C, N, H, W = x.size()
        x = x.transpose(1,2).contiguous().view(B*N, C, H, W)
        risk_factors_per_img =  (lambda N, risk_factors: [factor.expand( [N, *factor.size()]).contiguous().view([-1, factor.size()[-1]]).contiguous() for factor in risk_factors])(N, risk_factors) if risk_factors is not None else None
        # _, img_x, _ = self.image_encoder(x, risk_factors_per_img, batch)
        _, img_x, _ = self.image_encoder(x, risk_factors_per_img, batch, 
                                         img_index, y_l, y_u, x_l, x_u)
        # TODO: Uncomment
        img_x = img_x.view(B, N, -1)
        # print("img_x.size():", img_x.size())
        # print("self.image_repr_dim", self.image_repr_dim)
        img_x = img_x[:,:,: self.image_repr_dim]
        # print("img_x.size():", img_x.size())
        logit, transformer_hidden, activ_dict = self.transformer(img_x, risk_factors, batch)
        return logit, transformer_hidden, activ_dict