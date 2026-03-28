"""                                                                                                    
05_pytorch_classification.py                                                                           
Task 1: Predict whether a package will be DIM-flagged (binary classification)                          
using a PyTorch Lightning FFNN.                                                                        
Uses SCALED parquets. Evaluates with accuracy, precision, recall, F1, ROC AUC.                         
"""                                                                                                    
                                                                                                        
import pandas as pd                                                                                  
import numpy as np
import torch                                                                                           
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset                                                 
import pytorch_lightning as L                                                                        
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger                                                
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,                                
                            ConfusionMatrixDisplay)                                                 
import matplotlib.pyplot as plt                                                                        
from pathlib import Path                                                                               

ROOT = Path(__file__).resolve().parent.parent                                                          
                                                                                                    
# ----- 1. Load Data --------------------------------------------------------                          

train_df = pd.read_parquet(ROOT / 'data/train_scaled.parquet')                                         
val_df   = pd.read_parquet(ROOT / 'data/val_scaled.parquet')                                         
test_df  = pd.read_parquet(ROOT / 'data/test_scaled.parquet')                                          
                                                                                                        
# same target/feature split as all other scripts
target_cols  = ['dim_flag', 'log_net_charge', 'Net Charge Billed Currency']                            
feature_cols = [c for c in train_df.columns if c not in target_cols]                                   

X_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32)                             
X_val   = torch.tensor(val_df[feature_cols].values,   dtype=torch.float32)                           
X_test  = torch.tensor(test_df[feature_cols].values,   dtype=torch.float32)                            
                                                                                                        
# unsqueeze(1) makes shape (N,) -> (N,1) to match BCEWithLogitsLoss expectation                        
y_train = torch.tensor(train_df['dim_flag'].values, dtype=torch.float32).unsqueeze(1)                  
y_val   = torch.tensor(val_df['dim_flag'].values,   dtype=torch.float32).unsqueeze(1)                  
y_test  = torch.tensor(test_df['dim_flag'].values,   dtype=torch.float32).unsqueeze(1)                 

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)               
val_loader   = DataLoader(TensorDataset(X_val, y_val),     batch_size=512)                           
test_loader  = DataLoader(TensorDataset(X_test, y_test),   batch_size=512)                             
                                                                                                    
print(f"Features: {len(feature_cols)}")                                                                
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")                            
                                                                                                        
# class imbalance: 59.3% DIM=N vs 40.7% DIM=Y                                                          
# pos_weight tells the loss to upweight DIM=Y samples by this factor                                   
# formula: num_negatives / num_positives ≈ 0.593 / 0.407 ≈ 1.46                                        
n_pos = y_train.sum().item()                                                                           
n_neg = len(y_train) - n_pos                                                                           
pos_weight = torch.tensor([n_neg / n_pos])                                                             
print(f"Class balance — DIM=N: {n_neg:.0f} | DIM=Y: {n_pos:.0f} | pos_weight: {pos_weight.item():.2f}")
                                                                                                        
# ----- 2. Model ------------------------------------------------------------
                                                                                                        
class DIMClassifier(L.LightningModule):                                                                
    def __init__(self, input_dim, pos_weight, lr=1e-3):
        super().__init__()                                                                             
        self.save_hyperparameters()                                                                  
                                                                                                        
        # same 128→64→32→1 architecture as the regressor                                               
        # output is a raw logit (no sigmoid) — BCEWithLogitsLoss applies sigmoid internally            
        self.net = nn.Sequential(                                                                      
            nn.Linear(input_dim, 128),                                                               
            nn.BatchNorm1d(128),                                                                       
            nn.ReLU(),                                                                               
            nn.Dropout(0.3),
                                                                                                        
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),                                                                        
            nn.ReLU(),                                                                               
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
                                                                                                        
            nn.Linear(32, 1)   # single logit output
        )                                                                                              
                                                                                                    
        # BCEWithLogitsLoss = sigmoid + binary cross-entropy in one numerically stable op              
        # pos_weight scales the loss for positive (DIM=Y) samples to handle imbalance
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)                                     
                                                                                                        
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
        # AdamW with weight decay for regularization
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)        
        # cosine annealing decays LR smoothly from lr→0 over T_max epochs                              
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)                    
        return [optimizer], [scheduler]                                                                
                                                                                                        
# ----- 3. Train -------------------------------------------------------------                         

model = DIMClassifier(input_dim=len(feature_cols), pos_weight=pos_weight, lr=1e-3)                     
                                                                                                    
callbacks = [                                                                                          
    # stop if val_loss doesn't improve for 10 epochs
    EarlyStopping(monitor='val_loss', patience=10, mode='min'),                                        
    ModelCheckpoint(                                                                                 
        dirpath=str(ROOT / 'models'),                                                                  
        filename='pytorch_classifier_best',
        monitor='val_loss',                                                                            
        mode='min',                                                                                  
        save_top_k=1,                                                                                  
    )
]                                                                                                      
                                                                                                    
logger = TensorBoardLogger(save_dir=str(ROOT / 'logs'), name='classification')                         

trainer = L.Trainer(                                                                                   
    max_epochs=100,                                                                                  
    callbacks=callbacks,
    logger=logger,
    enable_progress_bar=True,
    deterministic=True
)                                                                                                      

trainer.fit(model, train_loader, val_loader)                                                           
                                                                                                    
# ----- 4. Evaluate on Validation Set ----------------------------------------

best_model = DIMClassifier.load_from_checkpoint(                                                       
    callbacks[1].best_model_path,
    input_dim=len(feature_cols),                                                                       
    pos_weight=pos_weight                                                                            
)                                                                                                      
best_model.eval()                                                                                    
best_model.cpu()   # move weights to CPU so we can pass CPU tensors directly
                                                                                                        
with torch.no_grad():
    # raw logits — not probabilities yet                                                               
    val_logits = best_model(X_val).squeeze()                                                           
    # sigmoid converts logits → probabilities between 0 and 1
    val_probs = torch.sigmoid(val_logits).numpy()                                                      
                                                                                                    
# default threshold: 0.5                                                                               
y_val_pred = (val_probs >= 0.5).astype(int)                                                          
y_val_true = val_df['dim_flag'].values                                                                 

accuracy  = accuracy_score(y_val_true, y_val_pred)                                                     
precision = precision_score(y_val_true, y_val_pred)                                                  
recall    = recall_score(y_val_true, y_val_pred)                                                       
f1        = f1_score(y_val_true, y_val_pred)                                                         
roc_auc   = roc_auc_score(y_val_true, val_probs)  # AUC uses probabilities, not hard labels            
                                                                                                        
print("\n=== PyTorch Classification (Validation Set) ===")                                             
print(f"Accuracy:  {accuracy:.4f}")                                                                    
print(f"Precision: {precision:.4f}")                                                                   
print(f"Recall:    {recall:.4f}")                                                                    
print(f"F1:        {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")                                                                     

# confusion matrix — bottom-left (False Neg) is the dangerous cell                                     
# false negatives = DIM=Y packages predicted as N → surprise cost from FedEx                         
cm = confusion_matrix(y_val_true, y_val_pred)                                                          
disp = ConfusionMatrixDisplay(cm, display_labels=['DIM=N', 'DIM=Y'])                                   
disp.plot(cmap='Blues')                                                                                
plt.title('PyTorch FFNN — Confusion Matrix (Val)')                                                     
plt.tight_layout()                                                                                   
plt.savefig(ROOT / 'figures/pytorch_cls_confusion.png', dpi=150)                                       
plt.show()                                                                                           
                                                                                                        
# ----- 5. Evaluate on Test Set -----------------------------------------------                      
                                                                                                        
with torch.no_grad():                                                                                
    test_logits = best_model(X_test).squeeze()
    test_probs  = torch.sigmoid(test_logits).numpy()                                                   

y_test_pred = (test_probs >= 0.5).astype(int)                                                          
y_test_true = test_df['dim_flag'].values                                                             

acc_test  = accuracy_score(y_test_true, y_test_pred)                                                   
prec_test = precision_score(y_test_true, y_test_pred)
rec_test  = recall_score(y_test_true, y_test_pred)                                                     
f1_test   = f1_score(y_test_true, y_test_pred)                                                         
auc_test  = roc_auc_score(y_test_true, test_probs)
                                                                                                        
print("\n=== PyTorch Classification (Test Set) ===")                                                 
print(f"Accuracy:  {acc_test:.4f}")                                                                    
print(f"Precision: {prec_test:.4f}")                                                                 
print(f"Recall:    {rec_test:.4f}")
print(f"F1:        {f1_test:.4f}")                                                                     
print(f"ROC AUC:   {auc_test:.4f}")