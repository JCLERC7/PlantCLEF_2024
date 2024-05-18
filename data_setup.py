import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset

class MyTrainDataset(Dataset):
    def __init__(self, datapath: str, pic_size=(224, 224), batch: int=64):
        self.datapath = datapath
        self.pic_size = pic_size
        self.batch = batch
      
        TRAIN_PERCENT = 0.9
        transform = transforms.Compose([
            # Resize the images to 224x224
            transforms.Resize(size=(224,224)),
            # Flip the images randomly on the horizontal
            transforms.RandomHorizontalFlip(p=0.5),  # p = probability of flip, 0.5 = 50% chance
            # Add your custom augmentation here
            transforms.RandomApply([transforms.TrivialAugmentWide(num_magnitude_bins=31)], p=0.5),
            # Turn the image into a torch.Tensor
            transforms.ToTensor(),
            # Normalize the image
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.data = datasets.ImageFolder(self.datapath,
                                         transform=transform,
                                         target_transform=None)

        self.train_size = int(TRAIN_PERCENT * len(self.data))
        self.test_size = len(self.data) - self.train_size
        self.train_data, self.test_data = torch.utils.data.random_split(self.data,
                                                                        [self.train_size, self.test_size])
        self.classes_name = self.data.classes
        
    def get_train_dataloader(self, num_workers=1):
        return DataLoader(self.train_data,
                          batch_size=self.batch,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True,
                          sampler=DistributedSampler(self.train_data)
                          )
    
    def get_test_dataloader(self, num_workers=1):
        return DataLoader(self.test_data,
                          batch_size=self.batch,
                          shuffle=False,
                          num_workers=num_workers,
                          pin_memory=True,
                          sampler=DistributedSampler(self.test_data)
                          )
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample, label = self.data[index][0], self.data[index][1]
        return sample, label


def Combine_datasets(root:str, transform, light_dataset):
    datasets_list = []

    for root_folder in os.listdir(root):
        root_path = os.path.join(root, root_folder)
        print(root_path)
        for folder_name in os.listdir(root_path):
            if folder_name in light_dataset:
                folder_path = os.path.join(root_path, folder_name)
                if os.path.isdir(folder_path):
                    print(folder_path)
                    dataset = datasets.ImageFolder(root=folder_path, transform=transform)
                    datasets_list.append(dataset)
        
    return datasets_list


class CustomConcatDataset(Dataset):
    def __init__(self, datasets, batch:int=12):
        TRAIN_PERCENT = 0.9
        self.datasets = datasets
        self.batch = batch
        self.cumulative_sizes = [0] + [len(d) for d in datasets]
        self.class_to_index_map = self._create_class_to_index_map()
        self.classes = self._get_classes()
        self.combined_dataset = ConcatDataset(datasets)
        self.train_size = int(TRAIN_PERCENT * len(self.combined_dataset))
        self.test_size = len(self.combined_dataset) - self.train_size
        self.train_data, self.test_data = torch.utils.data.random_split(self.combined_dataset,
                                                                        [self.train_size, self.test_size])
         
        
    def _create_class_to_index_map(self):
        class_to_index_map = {}
        current_index = 0
        for dataset in self.datasets:
            classes = dataset.classes
            for class_name in classes:
                class_to_index_map[class_name] = current_index
                current_index += 1
        return sorted(class_to_index_map.items(), key=lambda x:x[1])
    
    def _get_classes(self):
        all_classes = []
        for dataset in self.datasets:
            all_classes.extend(dataset.classes)
        return sorted(set(all_classes))
    
    def __len__(self):
        return sum(len(d) for d in self.datasets)
    
    def __getitem__(self, index):
        dataset_index = 0
        while index >= self.cumulative_sizes[dataset_index + 1]:
            dataset_index += 1
        return self.datasets[dataset_index][index - self.cumulative_sizes[dataset_index]]
    
    def class_to_index(self, class_name):
        return self.class_to_index_map[class_name]
    
    def get_train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch,
                          shuffle=False,
                        #   num_workers=num_workers,
                          pin_memory=True,
                          sampler=DistributedSampler(self.train_data)
                          )
    
    def get_test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.batch,
                          shuffle=False,
                        #   num_workers=num_workers,
                          pin_memory=True,
                          sampler=DistributedSampler(self.test_data)
                          )
    
    



# NUM_WORKERS = os.cpu_count()

# # def download_dataset(train_dir: str,
# #                      test_dir: str):
# #   urls = []
# #   file_paths = []
# #   urls.append("https://lab.plantnet.org/LifeCLEF/PlantCLEF2022/train/trusted/PlantCLEF2022_trusted_training_metadata.csv")
# #   urls.append("https://lab.plantnet.org/LifeCLEF/PlantCLEF2022/train/web/PlantCLEF2022_web_training_metadata.csv")
# #   urls.append("https://lab.plantnet.org/LifeCLEF/PlantCLEF2022/test/PlantCLEF2022_test_metadata.csv")

# #   script_path = os.path.dirname(os.path.abspath("Master_code.ipynb"))
# #   csv_path = Path(os.path.join(script_path, "PlantCLEF_CSV"))

# #   if csv_path.is_dir():
# #         print(f"{csv_path} directory exists.")
# #   else:
# #     print(f"Did not find {csv_path} directory, creating one...")
# #     csv_path.mkdir(parents=True, exist_ok=True)
    
# #   for url in urls:
# #     filename = url.split("/")[-1]
# #     file_path = os.path.join(csv_path, filename)
# #     file_paths.append(file_path)
    
# #     if not(os.path.exists(file_path)):
# #         wget.download(url, file_path)
# #         print("Download the file")

# def create_dataloaders(
#     dataset_dir: str, 
#     transform: transforms.Compose,
#     batch_size: int = 64, 
#     num_workers: int=NUM_WORKERS,
#     train_procent: int=0.9,
# ):
#   """Creates training and testing DataLoaders.

#   Takes in a training directory and testing directory path and turns
#   them into PyTorch Datasets and then into PyTorch DataLoaders.

#   Args:
#     train_dir: Path to training directory.
#     test_dir: Path to testing directory.
#     transform: torchvision transforms to perform on training and testing data.
#     batch_size: Number of samples per batch in each of the DataLoaders.
#     num_workers: An integer for number of workers per DataLoader.

#   Returns:
#     A tuple of (train_dataloader, test_dataloader, class_names).
#     Where class_names is a list of the target classes.
#     Example usage:
#       train_dataloader, test_dataloader, class_names = \
#         = create_dataloaders(train_dir=path/to/train_dir,
#                              test_dir=path/to/test_dir,
#                              transform=some_transform,
#                              batch_size=32,
#                              num_workers=4)
#   """
#   # Use ImageFolder to create dataset(s)
#   dataset = datasets.ImageFolder(dataset_dir,
#                                     transform=transform,
#                                     target_transform=None)
  
#   train_size = int(train_procent * len(dataset))
#   test_size = len(dataset) - train_size
#   train_data, test_data = torch.utils.data.random_split(dataset,[train_size, test_size])

#   # Get class names
#   class_names = dataset.classes

#   # Turn images into data loaders
#   train_dataloader = DataLoader(
#       train_data,
#       batch_size=batch_size,
#       shuffle=True,
#       num_workers=num_workers,
#       pin_memory=True,
#   )
#   test_dataloader = DataLoader(
#       test_data,
#       batch_size=batch_size,
#       shuffle=False,
#       num_workers=num_workers,
#       pin_memory=True,
#   )

#   return train_dataloader, test_dataloader, class_names