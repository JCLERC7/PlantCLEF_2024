import timm
import torch

class vit_small_dinov2():
    
    def creat_model(nbr_classes):
        model = timm.create_model('vit_small_patch14_reg4_dinov2.lvd142m',
                        pretrained=True,
                        num_classes=0,
                        img_size=224)
        model.head = torch.nn.Sequential(torch.nn.Linear(model.num_features, nbr_classes), torch.nn.Sigmoid())
        return model