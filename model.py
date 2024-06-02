import timm
import torch

def vit_small_dinov2(nbr_classes:int):
    model = timm.create_model('vit_small_patch14_reg4_dinov2.lvd142m',
                          pretrained=True,
                          num_classes=0)
    model.head = torch.nn.Sequential(torch.nn.Linear(model.num_features, nbr_classes), torch.nn.Sigmoid())
    return model