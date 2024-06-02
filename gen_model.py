import timm
import torch

class vit_small_dinov2:
    def __init__(self, nbr_classes:int):
        self.nbr_classes = nbr_classes
        
        model = timm.create_model('vit_small_patch14_reg4_dinov2.lvd142m',
                          pretrained=True,
                          num_classes=0)
        model.head = torch.nn.Sequential(torch.nn.Linear(model.num_features, self.nbr_classes), torch.nn.Sigmoid())
        return model