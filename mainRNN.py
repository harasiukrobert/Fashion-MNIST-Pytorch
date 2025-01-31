import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.utils.data import DataLoader, random_split

# Hyper-parameters
torch.set_float32_matmul_precision('high')
input_size = 28
hidden_size = 256
num_layers = 2
batch_size = 64
num_epochs = 10
num_classes = 10
learning_rate = 0.001


class RNN(pl.LightningModule):
    def __init__(self):
        super(RNN, self).__init__()
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()

        out, _ = self.rnn(x, h0)
        out = out[:, -1 , :]
        out = self.linear(out)
        return out


    def prepare_data(self):
        torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
        torchvision.datasets.FashionMNIST(root='./data', train=False, download=True)

    def setup(self, stage=None):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        if stage == 'fit' or stage is None:
            full_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform)
            train_size = int(0.9 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

        if stage == 'test' or stage is None:
            self.test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                          persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                          persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                          persistent_workers=True, pin_memory=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def training_step(self, batch, batch_idx):
        criterion = nn.CrossEntropyLoss()
        images, labels = batch
        images = images.reshape(-1, 28, 28)
        outputs = self(images)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels, task="multiclass", num_classes=num_classes)
        self.log_dict({'train_loss': loss, 'train_acc': acc}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        criterion = nn.CrossEntropyLoss()
        images, labels = batch
        images = images.reshape(-1, 28, 28)
        outputs = self(images)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels, task="multiclass", num_classes=num_classes)
        self.log_dict({'val_loss': loss, 'val_acc': acc}, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        criterion = nn.CrossEntropyLoss()
        images, labels = batch
        images = images.reshape(-1, 28, 28)
        outputs = self(images)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels, task="multiclass", num_classes=num_classes)
        self.log_dict({'test_loss': loss, 'test_acc': acc}, on_epoch=True, prog_bar=True, logger=True)


if __name__ == '__main__':
    trainer = pl.Trainer(max_epochs=num_epochs, fast_dev_run=False)
    model = RNN()

    start_time = time.time()
    trainer.fit(model)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Model learning time: {elapsed_time:.2f} s")
    trainer.test(model)
