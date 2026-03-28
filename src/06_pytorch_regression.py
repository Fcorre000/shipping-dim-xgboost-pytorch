"""
06_pytorch_regression.py
Task 2: Predict net shipping charge (USD) using a PyTorch Lightning FFNN.
Uses SCALED parquets and log-transformed target (log_net_charge).
Evaluates in real dollars via np.expm1().
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ----- 1. Load Data ----------------------------------------------------

train_df = pd.read_parquet(ROOT / 'data/train_scaled.parquet')
val_df = pd.read_parquet(ROOT / 'data/val_scaled.parquet')
test_df = pd.read_parquet(ROOT / 'data/test_scaled.parquet')

target_cols = ['dim_flag', 'log_net_charge', 'Net Charge Billed Currency']
feature_cols = [c for c in train_df.columns if c not in target_cols]

X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)
X_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32)
X_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32)

y_train = torch.tensor(train_df['log_net_charge'].values, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(val_df['log_net_charge'].values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(test_df['log_net_charge'].values, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=512)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=512)

print(f"Features: {len(feature_cols)}")
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ------ 2. Model ----------------------------------------------------

class ShippingRegressor(L.LightningModule):
    def __init__(self, input_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64,32),
            nn.ReLU(),

            nn.Linear(32,1)
        )
        self.loss_fn = nn.HuberLoss(delta=1.0)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]
    
#------ 3. Train ----------------------------------------------------

model = ShippingRegressor(input_dim=len(feature_cols), lr=1e-3)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, mode='min'),
    ModelCheckpoint(
        dirpath=str(ROOT / 'models'),
        filename='pytorch_regressor_best',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
    )
]

logger= TensorBoardLogger(save_dir=str(ROOT / 'logs'), name='regression')

trainer = L.Trainer(
    max_epochs=100,
    callbacks=callbacks,
    logger=logger,
    enable_progress_bar=True,
    deterministic=True
)

trainer.fit(model, train_loader, val_loader)

#----- 4. Evaulate on validation set ----------------------------------------------------

best_model = ShippingRegressor.load_from_checkpoint(
    callbacks[1].best_model_path,
    input_dim=len(feature_cols)
)
best_model.eval()
best_model.cpu()

with torch.no_grad():
    y_val_pred_log = best_model(X_val).squeeze().numpy()

y_val_pred_dollars = np.expm1(y_val_pred_log)
y_val_actual_dollars = val_df['Net Charge Billed Currency'].values

mae  = mean_absolute_error(y_val_actual_dollars, y_val_pred_dollars)
rmse = np.sqrt(mean_squared_error(y_val_actual_dollars, y_val_pred_dollars))
r2   = r2_score(y_val_actual_dollars, y_val_pred_dollars)

print("\n=== PyTorch Regression (Validation Set) ===")
print(f"MAE:  ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"R²:   {r2:.4f}")

#----- 5. Evaluate on test set ----------------------------------------------------

with torch.no_grad():
    y_test_pred_log = best_model(X_test).squeeze().numpy()

y_test_pred_dollars = np.expm1(y_test_pred_log)
y_test_actual_dollars = test_df['Net Charge Billed Currency'].values

mae_test  = mean_absolute_error(y_test_actual_dollars, y_test_pred_dollars)
rmse_test = np.sqrt(mean_squared_error(y_test_actual_dollars, y_test_pred_dollars))
r2_test   = r2_score(y_test_actual_dollars, y_test_pred_dollars)

print("\n=== PyTorch Regression (Test Set) ===")
print(f"MAE:  ${mae_test:.2f}")
print(f"RMSE: ${rmse_test:.2f}")
print(f"R²:   {r2_test:.4f}")