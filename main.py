"""
Trains a PyTorch image classification model using device-agnostic code.
"""
import argparse
import os
import torch
import data_setup, engine, utils, gen_model
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
def main (data_dir: str,
          epochs: int,
          batch_size: int,
          lr: float,
          save_every: int,
          snapshot_path: str,
          num_workers: int,
          fully_trained_model: str):
    
    # Function to set the seed for reproducibility (default seed = 42)
    utils.set_seed()
    
    # Setup target device
    ddp_setup()
    
    nbr_classes = len(os.listdir(data_dir))
    
    dataloader = data_setup.Dataloader_Gen(data_dir=data_dir, pic_size=(224, 224), batch=batch_size, num_worker=num_workers)
    
    train_dataloader = dataloader.get_train_dataloader
    test_dataloader = dataloader.get_test_dataloader
    valid_dataloader = dataloader.get_validation_dataloader
    
    cid_to_spid = utils.load_class_mapping("models/pretrained_models/class_mapping.txt")
    spid_to_sp = utils.load_species_mapping("models/pretrained_models/species_id_to_name.txt")
    
    writer = utils.create_writer("Final_model_training",
                                 "vit_small_patch14_reg4_dinov2",
                                 f"lr-{lr}_epoch-{epochs}_batch-{batch_size}_full_dataset")
    
    loss_fn = torch.nn.BCELoss()
    
    model_creator = gen_model.vit_small_dinov2
    model = model_creator.creat_model(nbr_classes=nbr_classes)

    optimizer = optim.Adam(model.parameters(), lr=lr) # 8.0e-05
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
    
    trainer = engine.Trainer(model=model,
                             train_data=train_dataloader,
                             test_data=test_dataloader,
                             valid_data=valid_dataloader,
                             optimizer=optimizer,
                             save_every=save_every,
                             snapshot_path=snapshot_path,
                             loss_fn=loss_fn,
                             scheduler=scheduler,
                             writer=writer)
    
    trainer.train(max_epochs=epochs)
    destroy_process_group()

    # # Save the model with help from utils.py
    # utils.save_model(model=model,
    #                 target_dir="models/final_models",
    #                 model_name=fully_trained_model)
    
if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Simple example of training script using Dino.")
    parser.add_argument("-p", "--data_dir", required=False, type=str, default="data/PlantCLEF2022_Training", help="The data folder on disk")
    parser.add_argument("-e", "--epochs", required=False, type=int, default=10, help="The number of training Epochs")
    parser.add_argument("--batch", required=False, type=int, default=48, help="The size of the batch")
    parser.add_argument("--lr", required=False, type=float, default=8.0e-05, help="The learning rate used for the training")
    parser.add_argument("--save_every", required=False, type=int, default=2, help="How often the model is saved per epochs during the trainning")
    parser.add_argument("--snapshot_path", required=False, type=str, default="models/snapshot/snapshot.pt", help="File location of the intermadiate saved model")
    parser.add_argument("--num_workers", required=False, type=int, choices=[0, 1, 2, 3, 4, 5], default=2, help="Number of process running")
    parser.add_argument("--fully_trained_model", required=False, type=str, default="Small_Dinov2_trained_Vx.pth")
    args = parser.parse_args()
    
    main(data_dir=args.data_dir,
         epochs=args.epochs,
         batch_size=args.batch,
         lr=args.lr,
         save_every=args.save_every,
         snapshot_path=args.snapshot_path,
         num_workers=args.num_workers,
         fully_trained_model=args.fully_trained_model)