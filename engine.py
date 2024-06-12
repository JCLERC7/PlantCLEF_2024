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
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
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

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_fn(output, targets)
        train_pred_labels = torch.round(output)
        loss.backward()
        self.optimizer.step()
        return loss.item(), ((train_pred_labels == targets).sum().item()/torch.numel(train_pred_labels))
        
        
    def _test_batch(self, source, targets):
        test_output = self.model(source)
        loss = self.loss_fn(test_output, targets)
        test_pred_labels = torch.round(test_output)
        return loss.item(), ((test_pred_labels == targets).sum().item()/torch.numel(test_pred_labels))
        

    def _run_epoch(self, epoch):
        train_loss, train_acc = 0, 0
        test_loss, test_acc = 0, 0
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)} | Time: {datetime.now()}")
        self.train_data.sampler.set_epoch(epoch)
        self.test_data.sampler.set_epoch(epoch)
        # Run one batch at the time
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
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
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch,
        
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
        # Close the writer
        self.writer.close()
