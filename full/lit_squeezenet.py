import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utils.gazetrack_data import gazetrack_dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

class eye_model(nn.Module):
  def __init__(self):
    super(eye_model, self).__init__()

    self.block1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, padding='same', kernel_size=1, bias=True),
        nn.ReLU(),
    )

    self.fire1_0, self.fire_1_1, self.fire1_2 = self.squeezenet_fire(32, 24, 64)

    self.max_pool1 = nn.MaxPool2d(kernel_size=2)

    self.fire2_0, self.fire_2_1, self.fire2_2 = self.squeezenet_fire(64, 48, 256)

    self.max_pool2 = nn.MaxPool2d(kernel_size=2)

    self.fire3_0, self.fire_3_1, self.fire3_2 = self.squeezenet_fire(256, 128, 512)

    self.global_avg = nn.AvgPool2d(kernel_size=32)
    
  def squeezenet_fire(self, in_channels, squeeze, expand, bnmomemtum=0.9):
    y = nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=squeeze, kernel_size=1, padding='same'),
      nn.ReLU(),
      nn.BatchNorm2d(squeeze, momentum=bnmomemtum)
    )

    y1 = nn.Sequential(
      nn.Conv2d(in_channels=squeeze, out_channels=expand//2, kernel_size=1, padding='same'),
      nn.ReLU(),
      nn.BatchNorm2d(expand//2, momentum=bnmomemtum)
    )

    y3 = nn.Sequential(
      nn.Conv2d(in_channels=expand//2, out_channels=expand//2, kernel_size=1, padding='same'),
      nn.ReLU(),
      nn.BatchNorm2d(expand//2, momentum=bnmomemtum)
    )

    return y, y1, y3

  def forward(self, x):
    block1_out = self.block1(x)

    fire1_0_out = self.fire1_0(block1_out)
    fire1_1_out = self.fire1_1(fire1_0_out)
    fire1_2_out = self.fire1_2(fire1_1_out)
    fire1_out = torch.cat((fire1_1_out, fire1_2_out), 1)

    maxpool1_out = self.max_pool1(fire1_out)

    fire2_0_out = self.fire2_0(maxpool1_out)
    fire2_1_out = self.fire2_1(fire2_0_out)
    fire2_2_out = self.fire2_2(fire2_1_out)
    fire2_out = torch.cat((fire2_1_out, fire2_2_out), 1)

    maxpool2_out = self.max_pool2(fire2_out)

    fire3_0_out = self.fire3_0(maxpool2_out)
    fire3_1_out = self.fire3_1(fire3_0_out)
    fire3_2_out = self.fire3_2(fire3_1_out)
    fire3_out = torch.cat((fire3_1_out, fire3_2_out), 1)

    out = self.global_avg(fire3_out)

    return out
    
class landmark_model(nn.Module):
    def __init__(self):
        super(landmark_model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 128),
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
        self.combined_model = nn.Sequential(nn.Linear(512+512+16, 8),
                                            nn.BatchNorm1d(8, momentum=0.9),
                                            nn.Dropout(0.12),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(8, 4),
                                            nn.BatchNorm1d(4, momentum=0.9),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(4, 2),)        

    def forward(self, leftEye, rightEye, lms):
        l_eye_feat = torch.flatten(self.eye_model(leftEye), 1)
        r_eye_feat = torch.flatten(self.eye_model(rightEye), 1)
        
        lm_feat = self.lmModel(lms)
        
        combined_feat = torch.cat((l_eye_feat, r_eye_feat, lm_feat), 1)
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
        _, l_eye, r_eye, kps, y, _, _ = batch
        y_hat = self(l_eye, r_eye, kps)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.logger.experiment.log_metric('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        _, l_eye, r_eye, kps, y, _, _ = batch
        y_hat = self(l_eye, r_eye, kps)
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