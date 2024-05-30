"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import argparse
import os
import torch
import timm
import data_setup, engine, utils
import torch.optim
import torch.optim as optim
from torchvision import transforms
from torch.utils.data.distributed import DistributedDampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import pandas

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
# TODO: Changer les arguments !!!
def main (data_path: str = "data/PlantCLEF2022_Training",
          total_epochs: int = 5,
          batch_size: int = 8,
          lr: float = 0.001,
          save_every: int=2,
          snapshot_path: str = "load/snapshot.pt"):
    
    # Function to set the seed for reproducibility (default seed = 42)
    utils.set_seed()
    
    # Setup target device
    ddp_setup()
    
    dataloader = data_setup.Dataloader_Gen(data_dir=, pic_size=, batch=, num_worker=)
    
    training_dataloader = combine_dataset.get_train_dataloader()
    test_dataloader = combine_dataset.get_test_dataloader()
    loss_fn = torch.nn.CrossEntropyLoss()
    
    classes = combine_dataset.classes

    # Create model with help from model_builder.py
    # model = timm.create_model("eva02_base_patch14_448.mim_in22k_ft_in22k", pretrained=True, num_classes=len(class_names)).to(device)
    model = timm.create_model("eva02_tiny_patch14_224.mim_in22k", pretrained=True, num_classes=len(classes))
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    writer = engine.create_writer("Test", "Eva02_224_in22k")
    
    Trainer = engine.Trainer(model=model,
                             train_data=training_dataloader,
                             test_data=test_dataloader,
                             optimizer=optimizer,
                             save_every=save_every,
                             snapshot_path=snapshot_path,
                             loss_fn=loss_fn,
                             scheduler=scheduler,
                             writer=writer)
    
    Trainer.train(total_epochs)
    destroy_process_group()

    # Save the model with help from utils.py
    utils.save_model(model=model,
                    target_dir="models",
                    model_name="05_going_modular_script_mode_tinyvgg_model.pth")
    
if __name__ == "__main__":
    import sys
    # parser = argparse.ArgumentParser(description="Simple example of training script using timm.")
    # parser.add_argument("--data_dir", required=True, help="The data folder on disk")
    # parser.add_argument("--epochs", required=True, type=int, help="The number of training Epochs")
    # parser.add_argument("--batch", required=False, help="The size of the batch")
    # args = parser.parse_args()
    # main(data_path=args.data_dir, epochs=args.epochs, batch_size=args.batch)
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    main(save_every=save_every, total_epochs=total_epochs)