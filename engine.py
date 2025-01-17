"""
Contains functions for training and testing a PyTorch model.
"""
import os
import utils
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import accuracy_score

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        valid_data: DataLoader,
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
        self.valid_data = valid_data
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_every = save_every
        self.epochs_run = 0
        self.max_accuracy = 0
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
        train_pred_labels = torch.round(output)
        loss.backward()
        self.optimizer.step()
        return loss.item(), accuracy_score(y_true=targets.cpu().detach().numpy(), y_pred=train_pred_labels.cpu().detach().numpy(), normalize=True)
        
        
    def _test_batch(self, source, targets):
        test_output = self.model(source)
        loss = self.loss_fn(test_output, targets)
        test_pred_labels = torch.round(test_output)
        return loss.item(), accuracy_score(y_true=targets.cpu().detach().numpy(), y_pred=test_pred_labels.cpu().detach().numpy(), normalize=True)
        

    def _run_epoch(self, epoch):
        train_loss, train_acc = 0, 0
        test_loss, test_acc = 0, 0
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)} | Time: {datetime.now()}")
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
            
        if epoch % 2 == 0:
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
            
            if self.max_accuracy < test_acc:
                utils.save_model(model=self.model, target_dir="models/final_models", model_name="Small_Dinov2_trained_Vx.pth")
            
            if self.writer:
                self.writer.add_scalar("test/Loss", test_loss, epoch)
                self.writer.add_scalar("test/Accuracy", test_acc, epoch)
                
                
        if self.writer:
            self.writer.add_scalar("train/Loss", train_loss, epoch)
            self.writer.add_scalar("train/Accuracy", train_acc, epoch)
        else:
            pass

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
        
    def validate(self):
        self.model.eval()
        with torch.inference_mode():
            for source, targets in self.valid_data:
                source = source.to(self.local_rank)
                targets = targets.to(self.local_rank)
                loss, accu = self._test_batch(source, targets)
                valid_loss += loss
                valid_acc += accu
        valid_loss = valid_loss / len(self.valid_data)
        valid_acc = valid_acc / len(self.valid_data)
        
        if self.writer:
            self.writer.add_scalar("test/Loss", valid_loss)
            self.writer.add_scalar("test/Accuracy", valid_acc)
        # Close the writer
        self.writer.close()
