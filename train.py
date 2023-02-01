import torch
import argparse
import pytorch_lightning as pl
from model import CNN
from data import MNISTDataModule

def train(optim, batch_size):
    dm = MNISTDataModule('.', batch_size, 20)
    optim_dict = {'adam':torch.optim.Adam, 'nadam': torch.optim.NAdam, 'rmsprop': torch.optim.RMSprop}
    model = CNN(optim_dict[optim])

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        precision=16
    )

    trainer.fit(model, dm)
    trainer.test(model, dm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-bs', type=int, default=512, help='Batch size')
    parser.add_argument('--optimizer', '-o', type=str, default='adam', choices={'adam', 'nadam', 'rmsprop'})
    args = parser.parse_args()
    train(args.optimizer, args.batch_size)