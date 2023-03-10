import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils.gazetrack_data_one_eye import gazetrack_dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

class eye_model(nn.Module):
  def __init__(self):
    super(eye_model, self).__init__()

    self.block1 = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=0),
        nn.BatchNorm2d(32, momentum=0.9),
        nn.LeakyReLU(inplace=True),

        nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0),
        nn.BatchNorm2d(32, momentum=0.9),
        nn.LeakyReLU(inplace=True),

        nn.AvgPool2d(kernel_size=2),
        nn.Dropout(0.02)
    )

    self.block1_downsample = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0),
        nn.BatchNorm2d(64, momentum=0.9),
        # nn.LeakyReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2)
    )

    self.block2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(64, momentum=0.9),
        nn.LeakyReLU(inplace=True),

        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0),
        nn.BatchNorm2d(64, momentum=0.9),
        # nn.LeakyReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2),
        nn.Dropout(0.02)
    )

    self.block2_activation = nn.LeakyReLU(inplace=True)

    self.block2_downsample = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
        nn.BatchNorm2d(128, momentum=0.9),
        # nn.LeakyReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2),
    )

    self.block3 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0),
        nn.BatchNorm2d(128, momentum=0.9),
        nn.LeakyReLU(inplace=True),

        nn.Conv2d(128, 128, kernel_size=2, stride=2, padding=0),
        nn.BatchNorm2d(128, momentum=0.9),
        # nn.LeakyReLU(inplace=True),

        nn.AvgPool2d(kernel_size=2),
        nn.Dropout(0.02)
    )

    self.block3_activation = nn.LeakyReLU(inplace=True)

  def forward(self, x):
    block1_out = self.block1(x)
    block1_residual = self.block1_downsample(block1_out)

    block2_out = self.block2_activation(self.block2(block1_out) + block1_residual)
    block2_residual = self.block2_downsample(block2_out)

    block3_out = self.block3_activation(self.block3(block2_out) + block2_residual)
    
    return block3_out
    
class landmark_model(nn.Module):
    def __init__(self):
        super(landmark_model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 128),
            nn.BatchNorm1d(128, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(128, 16),
            nn.BatchNorm1d(16, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.BatchNorm1d(16, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class lit_gazetrack_model(pl.LightningModule):
    def __init__(self, data_path, save_path, batch_size, logger, workers=20):
        super(lit_gazetrack_model, self).__init__()
        
        self.lr = 0.016
        self.batch_size = batch_size
        self.data_path = data_path
        self.workers = workers
        print("Data path: ", data_path)
        self.save_path = save_path
        PARAMS = {'batch_size': self.batch_size,
                  'init_lr': self.lr,
                  'data_path': self.data_path,
                  'save_path': self.save_path,
                    'scheduler': "Plateau"}
        logger.log_hyperparams(PARAMS)
        
        self.eye_model = eye_model()
        self.lmModel = landmark_model()
        self.combined_model = nn.Sequential(nn.Linear(512+16, 8),
                                            nn.BatchNorm1d(8, momentum=0.9),
                                            nn.Dropout(0.12),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(8, 4),
                                            nn.BatchNorm1d(4, momentum=0.9),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(4, 2),)        

    def forward(self, eye, lms):
        eye_feat = torch.flatten(self.eye_model(eye), 1)
        
        lm_feat = self.lmModel(lms)
        
        combined_feat = torch.cat((eye_feat, lm_feat), 1)
        out = self.combined_model(combined_feat)
        return out
    
    def train_dataloader(self):
        train_dataset = gazetrack_dataset(self.data_path+"/train/", phase='train')
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.workers, shuffle=True)
        self.logger.log_hyperparams({'Num_train_files': len(train_dataset)})
        return train_loader
    
    def val_dataloader(self):
        dataVal = gazetrack_dataset(self.data_path+"/val/", phase='val')
        val_loader = DataLoader(dataVal, batch_size=self.batch_size, num_workers=self.workers, shuffle=False)
        self.logger.log_hyperparams({'Num_val_files': len(dataVal)})
        return val_loader
    
    def training_step(self, batch, batch_idx):
        _, eye, kps, y, _, _ = batch
        y_hat = self(eye, kps)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.logger.experiment.log_metric('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        _, eye, kps, y, _, _ = batch
        y_hat = self(eye, kps)
        val_loss = F.mse_loss(y_hat, y)
        self.logger.experiment.log_metric('val_loss', val_loss)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-07)
#         scheduler = ExponentialLR(optimizer, gamma=0.64, verbose=True)
        scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }