"""
Contains functions for training and testing a PyTorch model.
"""
import os
from typing import Dict, List, Tuple
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from tqdm.auto import tqdm
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        loss_fn: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler,
        writer: SummaryWriter
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.scheduler = scheduler
        self.writer = writer
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_fn(output, targets)
        train_pred_labels = output.argmax(dim=1)
        loss.backward()
        self.optimizer.step()
        return loss.item(), ((train_pred_labels == targets).sum().item()/len(train_pred_labels))
        
        
    def _test_batch(self, source, targets):
        test_output = self.model(source)
        loss = self.loss_fn(test_output, targets)
        test_pred_labels = test_output.argmax(dim=1)
        return loss.item(), ((test_pred_labels == targets).sum().item()/len(test_pred_labels))
        

    def _run_epoch(self, epoch):
        train_loss, train_acc = 0, 0
        test_loss, test_acc = 0, 0
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        self.test_data.sampler.set_epoch(epoch)
        # Run one batch at the time
        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            loss, accu = self._run_batch(source, targets)
            train_loss += loss
            train_acc += accu
        self.scheduler.step()
        train_loss = train_loss / len(self.train_data)
        train_acc = train_acc / len(self.train_data)
            
        if epoch % 5 == 0 and epoch != 0:
            self.model.eval()
            with torch.inference_mode():
                for source, targets in self.test_data:
                    source = source.to(self.local_rank)
                    targets = targets.to(self.local_rank)
                    loss, accu = self._test_batch(source, targets)
                    test_loss += loss
                    test_acc += accu
            test_loss = test_loss / len(self.test_data)
            test_acc = test_acc / len(self.test_data)
            
        if self.writer:
            self.writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                               global_step=epoch)
            self.writer.add_scalars(main_tag="Accuracy", 
                               tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc}, 
                               global_step=epoch)
        else:
            pass

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
        # Close the writer
        self.writer.close()
    
def create_writer( experiment_name: str, model_name: str, extra: str=None):
    timestamp = datetime.now().strftime("%Y-%m-%d")
    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
    
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

# def train_step(model, 
#                dataloader: torch.utils.data.DataLoader, 
#                loss_fn: torch.nn.Module, 
#                optimizer: torch.optim.Optimizer,
#                device: torch.device,
#                scheduler: torch.optim.lr_scheduler) -> Tuple[float, float]:
#     """Trains a PyTorch model for a single epoch.

#     Turns a target PyTorch model to training mode and then
#     runs through all of the required training steps (forward
#     pass, loss calculation, optimizer step).

#     Args:
#     model: A PyTorch model to be trained.
#     dataloader: A DataLoader instance for the model to be trained on.
#     loss_fn: A PyTorch loss function to minimize.
#     optimizer: A PyTorch optimizer to help minimize the loss function.
#     device: A target device to compute on (e.g. "cuda" or "cpu").

#     Returns:
#     A tuple of training loss and training accuracy metrics.
#     In the form (train_loss, train_accuracy). For example:

#     (0.1112, 0.8743)
#     """
#     # Put model in train mode
#     model.train()

#     # Setup train loss and train accuracy values
#     train_loss, train_acc = 0, 0

#     # Loop through data loader data batches
#     for batch, (X, y) in enumerate(dataloader):
#         # Send data to target device
#         X, y = X.to(device), y.to(device)

#         # 1. Forward pass
#         y_pred = model(X)

#         # 2. Calculate  and accumulate loss
#         loss = loss_fn(y_pred, y)
#         train_loss += loss.item() 

#         # 3. Optimizer zero grad
#         optimizer.zero_grad()

#         # 4. Loss backward
#         loss.backward()

#         # 5. Optimizer step
#         optimizer.step()

#         # Calculate and accumulate accuracy metric across all batches
#         y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
#         train_acc += (y_pred_class == y).sum().item()/len(y_pred)

#     # Adjust metrics to get average loss and accuracy per batch 
#     train_loss = train_loss / len(dataloader)
#     scheduler.step()
#     train_acc = train_acc / len(dataloader)
#     return train_loss, train_acc

# def test_step(model: torch.nn.Module, 
#               dataloader: torch.utils.data.DataLoader, 
#               loss_fn: torch.nn.Module,
#               device: torch.device) -> Tuple[float, float]:
#     """Tests a PyTorch model for a single epoch.

#     Turns a target PyTorch model to "eval" mode and then performs
#     a forward pass on a testing dataset.

#     Args:
#     model: A PyTorch model to be tested.
#     dataloader: A DataLoader instance for the model to be tested on.
#     loss_fn: A PyTorch loss function to calculate loss on the test data.
#     device: A target device to compute on (e.g. "cuda" or "cpu").

#     Returns:
#     A tuple of testing loss and testing accuracy metrics.
#     In the form (test_loss, test_accuracy). For example:

#     (0.0223, 0.8985)
#     """
#     # Put model in eval mode
#     model.eval() 

#     # Setup test loss and test accuracy values
#     test_loss, test_acc = 0, 0

#     # Turn on inference context manager
#     with torch.inference_mode():
#         # Loop through DataLoader batches
#         for batch, (X, y) in enumerate(dataloader):
#             # Send data to target device
#             X, y = X.to(device), y.to(device)

#             # 1. Forward pass
#             test_pred_logits = model(X)

#             # 2. Calculate and accumulate loss
#             loss = loss_fn(test_pred_logits, y)
#             test_loss += loss.item()

#             # Calculate and accumulate accuracy
#             test_pred_labels = test_pred_logits.argmax(dim=1)
#             test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

#     # Adjust metrics to get average loss and accuracy per batch 
#     test_loss = test_loss / len(dataloader)
#     test_acc = test_acc / len(dataloader)
#     return test_loss, test_acc

# def train(model: torch.nn.Module, 
#           train_dataloader: torch.utils.data.DataLoader, 
#           test_dataloader: torch.utils.data.DataLoader, 
#           optimizer: torch.optim.Optimizer,
#           loss_fn: torch.nn.Module,
#           epochs: int,
#           device: torch.device,
#           scheduler: torch.optim.lr_scheduler) -> Dict[str, List[float]]:
#     """Trains and tests a PyTorch model.

#     Passes a target PyTorch models through train_step() and test_step()
#     functions for a number of epochs, training and testing the model
#     in the same epoch loop.

#     Calculates, prints and stores evaluation metrics throughout.

#     Args:
#     model: A PyTorch model to be trained and tested.
#     train_dataloader: A DataLoader instance for the model to be trained on.
#     test_dataloader: A DataLoader instance for the model to be tested on.
#     optimizer: A PyTorch optimizer to help minimize the loss function.
#     loss_fn: A PyTorch loss function to calculate loss on both datasets.
#     epochs: An integer indicating how many epochs to train for.
#     device: A target device to compute on (e.g. "cuda" or "cpu").

#     Returns:
#     A dictionary of training and testing loss as well as training and
#     testing accuracy metrics. Each metric has a value in a list for 
#     each epoch.
#     In the form: {train_loss: [...],
#               train_acc: [...],
#               test_loss: [...],
#               test_acc: [...]} 
#     For example if training for epochs=2: 
#              {train_loss: [2.0616, 1.0537],
#               train_acc: [0.3945, 0.3945],
#               test_loss: [1.2641, 1.5706],
#               test_acc: [0.3400, 0.2973]} 
#     """
#     # Create empty results dictionary
#     results = {"train_loss": [],
#                "train_acc": [],
#                "test_loss": [],
#                "test_acc": []
#     }

#     # Loop through training and testing steps for a number of epochs
#     for epoch in tqdm(range(epochs)):
#         train_loss, train_acc = train_step(model=model,
#                                           dataloader=train_dataloader,
#                                           loss_fn=loss_fn,
#                                           optimizer=optimizer,
#                                           device=device,
#                                           scheduler=scheduler)
        
#         test_loss, test_acc = test_step(model=model,
#                                         dataloader=test_dataloader,
#                                         loss_fn=loss_fn,
#                                         device=device)

#         # Print out what's happening
#         print(
#           f"Epoch: {epoch+1} | "
#           f"train_loss: {train_loss:.4f} | "
#           f"train_acc: {train_acc:.4f} | "
#           f"test_loss: {test_loss:.4f} | "
#           f"test_acc: {test_acc:.4f}"
#         )

#         # Update results dictionary
#         results["train_loss"].append(train_loss)
#         results["train_acc"].append(train_acc)
#         results["test_loss"].append(test_loss)
#         results["test_acc"].append(test_acc)

#     # Return the filled results at the end of the epochs
#     return results
# 
