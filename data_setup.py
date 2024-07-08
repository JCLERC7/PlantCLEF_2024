import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

class Dataloader_Gen(Dataset):
    def __init__(self, data_dir: str, num_worker: int, pic_size=(518, 518), batch: int=64):
        self.data_dir = data_dir
        self.pic_size = pic_size
        self.batch = batch
        self.nbr_classes = len(os.listdir(self.data_dir))
        
        TRAIN_PERCENT = 0.8
        
        data_transform = transforms.Compose([
            # Resize the images to 518x518
            transforms.Resize(pic_size),
            # Flip the images randomly on the horizontal
            transforms.RandomHorizontalFlip(p=0.5),  # p = probability of flip, 0.5 = 50% chance
            # Add your custom augmentation here
            transforms.RandomApply([transforms.TrivialAugmentWide(num_magnitude_bins=31)], p=0.5),
            # Turn the image into a torch.Tensor
            transforms.ToTensor(),
            # Normalize the image
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        def one_hot(label_idx, nbr_classes):
            one_hot_tensor = torch.zeros(nbr_classes, dtype=torch.float32)
            one_hot_tensor.scatter_(0, torch.tensor(label_idx), value=1)
            return one_hot_tensor
        
        label_transform = transforms.Compose([transforms.Lambda(lambda y: one_hot(y, self.nbr_classes))])

        self.dataset = datasets.ImageFolder(self.data_dir,
                                            transform=data_transform,
                                            target_transform=label_transform)

        self.dataset_size = int(len(self.dataset))
        self.train_size = int(TRAIN_PERCENT * self.dataset_size)
        self.test_size = int((self.dataset_size - self.train_size) / 2)
        self.val_size = self.dataset_size - self.train_size - self.test_size
        
        self.train_data, self.test_data, self.validation_data = torch.utils.data.random_split(self.dataset,
                                                                                              [self.train_size, self.test_size, self.validation_size])
        
    def get_train_dataloader(self, num_workers=2):
        return DataLoader(self.train_data,
                          batch_size=self.batch,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True,
                          sampler=DistributedSampler(self.train_data))
    
    def get_test_dataloader(self, num_workers=2):
        return DataLoader(self.test_data,
                          batch_size=self.batch,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True,
                          sampler=DistributedSampler(self.test_data))
        
    def get_validation_dataloader(self, num_workers=2):
        return DataLoader(self.validation_data,
                          batch_size=self.batch,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True,
                          sampler=DistributedSampler(self.validation_data))
        
    
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        sample, label = self.dataset[index][0], self.dataset[index][1]
        return sample, label
