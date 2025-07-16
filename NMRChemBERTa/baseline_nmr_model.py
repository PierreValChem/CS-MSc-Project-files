"""
Baseline SMILES-to-NMR model - much simpler architecture
"""
import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class BaselineNMRModel(nn.Module):
    """Very simple baseline model for NMR prediction"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_atoms = config.model.max_atoms
        
        # Load ChemBERTa
        self.chemberta = AutoModel.from_pretrained(config.model.chemberta_name)
        self.chemberta_dim = self.chemberta.config.hidden_size
        
        # Freeze ChemBERTa
        for param in self.chemberta.parameters():
            param.requires_grad = False
        
        # Simple pooling - just average
        self.dropout = nn.Dropout(0.1)
        
        # Direct prediction heads - much simpler
        self.h_predictor = nn.Sequential(
            nn.Linear(self.chemberta_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.max_atoms)
        )
        
        self.c_predictor = nn.Sequential(
            nn.Linear(self.chemberta_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.max_atoms)
        )
        
        # Initialize with small weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, input_ids, attention_mask, **kwargs):
        # Get ChemBERTa embeddings
        outputs = self.chemberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Simple average pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        
        # Apply dropout
        mean_embeddings = self.dropout(mean_embeddings)
        
        # Predict
        h_shifts = self.h_predictor(mean_embeddings)  # (batch, max_atoms)
        c_shifts = self.c_predictor(mean_embeddings)  # (batch, max_atoms)
        
        # Stack
        nmr_shifts = torch.stack([h_shifts, c_shifts], dim=-1)  # (batch, max_atoms, 2)
        
        return {'nmr_shifts': nmr_shifts}


class SimpleNMRLoss(nn.Module):
    """Simple MSE loss for NMR"""
    def __init__(self, config):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, predictions, targets, masks):
        pred_shifts = predictions['nmr_shifts']
        target_shifts = targets['nmr_shifts']
        nmr_mask = masks['nmr_mask']
        
        # Simple MSE
        loss = self.mse(pred_shifts, target_shifts)
        
        # Apply mask
        masked_loss = loss * nmr_mask
        num_valid = nmr_mask.sum() + 1e-6
        
        total_loss = masked_loss.sum() / num_valid
        
        # Separate losses
        h_loss = masked_loss[:, :, 0].sum() / (nmr_mask[:, :, 0].sum() + 1e-6)
        c_loss = masked_loss[:, :, 1].sum() / (nmr_mask[:, :, 1].sum() + 1e-6)
        
        return {
            'total_loss': total_loss,
            'nmr_loss': total_loss,
            'h_nmr_loss': h_loss,
            'c_nmr_loss': c_loss
        }


# Quick training script for baseline
def train_baseline():
    from config import Config
    from nmr_dataset import create_data_loaders
    import numpy as np
    from tqdm import tqdm
    
    # Config for baseline
    config = Config.from_yaml('config_smiles_nmr.yaml')
    config.data.batch_size = 32
    config.training.learning_rate = 1e-4
    config.data.max_files_limit = 15000  # Quick test
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, _, _ = create_data_loaders(config)
    
    # Create model
    model = BaselineNMRModel(config).to(device)
    loss_fn = SimpleNMRLoss(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Quick training loop
    for epoch in range(10):
        # Train
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            nmr_targets = torch.stack([
                batch['nmr_features']['h_shifts'],
                batch['nmr_features']['c_shifts']
            ], dim=-1).to(device)
            
            nmr_mask = torch.stack([
                batch['nmr_features']['h_mask'],
                batch['nmr_features']['c_mask']
            ], dim=-1).to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss_dict = loss_fn(outputs, {'nmr_shifts': nmr_targets}, {'nmr_mask': nmr_mask})
            loss = loss_dict['total_loss']
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validate
        model.eval()
        val_losses = []
        h_maes = []
        c_maes = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                nmr_targets = torch.stack([
                    batch['nmr_features']['h_shifts'],
                    batch['nmr_features']['c_shifts']
                ], dim=-1).to(device)
                
                nmr_mask = torch.stack([
                    batch['nmr_features']['h_mask'],
                    batch['nmr_features']['c_mask']
                ], dim=-1).to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss_dict = loss_fn(outputs, {'nmr_shifts': nmr_targets}, {'nmr_mask': nmr_mask})
                
                val_losses.append(loss_dict['total_loss'].item())
                
                # Calculate MAE
                pred = outputs['nmr_shifts']
                h_mae = torch.abs(pred[:, :, 0] - nmr_targets[:, :, 0])
                c_mae = torch.abs(pred[:, :, 1] - nmr_targets[:, :, 1])
                
                h_mae = (h_mae * nmr_mask[:, :, 0]).sum() / (nmr_mask[:, :, 0].sum() + 1e-6)
                c_mae = (c_mae * nmr_mask[:, :, 1]).sum() / (nmr_mask[:, :, 1].sum() + 1e-6)
                
                h_maes.append(h_mae.item())
                c_maes.append(c_mae.item())
        
        # Print results
        h_mae_ppm = np.mean(h_maes) * 2.07
        c_mae_ppm = np.mean(c_maes) * 50.26
        
        print(f"Epoch {epoch+1}: Train Loss: {np.mean(train_losses):.4f}, "
              f"Val Loss: {np.mean(val_losses):.4f}, "
              f"H MAE: {h_mae_ppm:.2f} ppm, C MAE: {c_mae_ppm:.2f} ppm")


if __name__ == "__main__":
    train_baseline()