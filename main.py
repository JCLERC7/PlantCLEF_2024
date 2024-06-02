"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import argparse
import os
import torch
import timm
import data_setup, engine, utils, model
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
def main (data_dir: str = "data/PlantCLEF2022_Training",
          total_epochs: int = 5,
          batch_size: int = 8,
          lr: float = 0.001,
          save_every: int=2,
          snapshot_path: str = "load/snapshot.pt"):
    
    # Function to set the seed for reproducibility (default seed = 42)
    utils.set_seed()
    
    # Setup target device
    ddp_setup()
    
    nbr_classes = len(os.listdir(data_dir))
    
    dataloader = data_setup.Dataloader_Gen(data_dir=data_dir, pic_size=(518, 518), batch=64, num_worker=2)
    
    train_dataloader = dataloader.get_train_dataloader
    test_dataloader = dataloader.get_test_dataloader
    valid_dataloader = dataloader.get_validation_dataloader
    
    cid_to_spid = utils.load_class_mapping("models/pretrained_models/class_mapping.txt")
    spid_to_sp = utils.load_species_mapping("models/pretrained_models/species_id_to_name.txt")
    
    writer = utils.create_writer("Run_3",
                                 "vit_small_patch14_reg4_dinov2",
                                 "lr-8.0e-05_epoch-100_batch-24_light_dataset")
    
    loss_fn = torch.nn.BCELoss()
    
    model = model.vit_small_dinov2(nbr_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=8.0e-05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
    
    trainer = engine.Trainer(model=model,
                             train_data=train_dataloader,
                             test_data=test_dataloader,
                             optimizer=optimizer,
                             save_every=2,
                             snapshot_path=snapshot_path,
                             loss_fn=loss_fn,
                             scheduler=scheduler,
                             writer=writer)
    
    trainer()
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