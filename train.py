import torch
import argparse
import pytorch_lightning as pl
from model import CNN
from data import MNISTDataModule
import wandb
wandb.login()

sweep_config = {
    'method': 'random'
    }
metric = {
    'name': 'val_loss',
    'goal': 'minimize'   
    }
parameters_dict = {
    'optimizer': {
        'values': ['adam', 'nadam','rmsprop']
        },
    # 'fc_layer_size': {
    #     'values': [128, 256, 512]
    #     },
    # 'dropout': {
    #       'values': [0.3, 0.4, 0.5]
    #     },
    }
parameters_dict.update({
'epochs': {
    'value': 10}
})
parameters_dict.update({
    'learning_rate': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'batch_size': {
        # integers between 32 and 256
        # with evenly-distributed logarithms 
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 512,
      }
    })
sweep_config['parameters'] = parameters_dict
sweep_config['metric'] = metric
sweep_id = wandb.sweep(sweep_config, project="Trial")



def train(optim=None, batch_size=None):
    run = wandb.init()
    config = wandb.config
    dm = MNISTDataModule('.', config.batch_size, 20)
    optim_dict = {'adam':torch.optim.Adam, 'nadam': torch.optim.NAdam, 'rmsprop': torch.optim.RMSprop}
    model = CNN(optim_dict[config.optimizer],config.learning_rate)

    trainer = pl.Trainer(
        max_epochs= config.epochs,
        accelerator="gpu",
        precision=16
    )

    trainer.fit(model, dm)
    trainer.test(model, dm)

if __name__ == '__main__':
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', '-bs', type=int, default=512, help='Batch size')
    # parser.add_argument('--optimizer', '-o', type=str, default='adam', choices={'adam', 'nadam', 'rmsprop'})
    # parser.add_argument('-f')
    # args = parser.parse_args()
    wandb.agent(sweep_id, train, count=5)