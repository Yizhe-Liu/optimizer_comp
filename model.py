import pytorch_lightning as pl
import torch
import torchvision
from torchmetrics.functional import accuracy
from torch import nn
import torch.nn.functional as F
import wandb
wandb.login()


class CNN(pl.LightningModule):
    def __init__(self, optim_dict,lr,optim_name):
        super().__init__()
        self.optim = optim_dict[optim_name]
        self.lr = lr
        self.resnet18 = torchvision.models.resnet18(num_classes=10)
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.maxpool = nn.Identity()
        self.optim_name = optim_name 
        
    def forward(self, x):
        return F.log_softmax(self.resnet18(x), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            wandb.log({f"{stage}_loss": loss})
            wandb.log({f"{stage}_Accuracy": acc})
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        if(self.optim_name == 'sgd'):
            return self.optim(self.parameters(), self.lr)
        else:
            return self.optim(self.parameters(), self.lr)