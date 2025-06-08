"""
Configuration management for NMR-ChemBERTa
"""

from dataclasses import dataclass
from typing import Optional, List
import yaml
import os


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Updated to latest ChemBERTa v2
    chemberta_name: str = 'DeepChem/ChemBERTa-77M-MLM'  # Latest version
    # Alternative options:
    # 'seyonec/ChemBERTa-zinc250k-v2-5M' (smaller)
    # 'seyonec/ChemBERTa-zinc250k-v2-77M' (medium)
    # 'seyonec/ChemBERTa-zinc250k-v2-177M' (larger)
    
    hidden_dim: int = 384
    num_atom_types: int = 10
    max_atoms: int = 200
    max_seq_length: int = 512
    dropout: float = 0.1
    num_attention_heads: int = 6
    freeze_chemberta: bool = True
    
    # Enhanced NMR encoding
    nmr_hidden_dim: int = 256  # Dedicated hidden dim for NMR features
    nmr_num_layers: int = 3  # More layers for NMR processing


@dataclass
class DataConfig:
    """Data processing configuration"""
    data_directory: str = "CSV_to_NMRe_output_v3/"
    batch_size: int = 16
    train_split: float = 0.7  # Changed from 0.8
    val_split: float = 0.2    # Changed from 0.1
    test_split: float = 0.1   # Added test split
    num_workers: int = 4
    pin_memory: bool = True
    max_files_limit: Optional[int] = 1000  # Limit for testing
    
    # Cross-validation settings
    use_cross_validation: bool = False
    num_folds: int = 5
    current_fold: int = 0


@dataclass
class TrainingConfig:
    """Training configuration"""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1
    save_every_n_epochs: int = 5
    validate_every_n_steps: int = 100
    early_stopping_patience: int = 10
    scheduler_patience: int = 5
    use_amp: bool = True
    
    # Enhanced loss weights for better NMR connection
    nmr_loss_weight: float = 2.0  # Increased from 1.0
    position_loss_weight: float = 1.0
    atom_type_loss_weight: float = 1.0
    smiles_position_loss_weight: float = 0.5
    
    # Additional loss settings
    nmr_loss_type: str = 'huber'  # 'mse', 'mae', 'huber'
    nmr_loss_reduction: str = 'weighted'  # 'mean', 'weighted'
    use_nmr_attention_loss: bool = True  # Additional attention-based loss


@dataclass
class HardwareConfig:
    """Hardware optimization configuration"""
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'
    mixed_precision: bool = True
    compile_model: bool = False  # PyTorch 2.0 compilation
    use_distributed: bool = False
    gradient_checkpointing: bool = False
    channels_last: bool = True  # Memory format optimization


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    log_level: str = 'INFO'
    use_wandb: bool = False
    use_tensorboard: bool = True
    log_dir: str = './logs'
    checkpoint_dir: str = './checkpoints'
    experiment_name: str = 'nmr_chemberta'
    
    # Visualization settings
    plot_every_n_epochs: int = 5
    save_plots: bool = True
    plot_dir: str = './plots'


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    hardware: HardwareConfig
    logging: LoggingConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            hardware=HardwareConfig(**config_dict.get('hardware', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )
    
    def to_yaml(self, config_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'hardware': self.hardware.__dict__,
            'logging': self.logging.__dict__
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def get_default_config() -> Config:
    """Get default configuration"""
    return Config(
        model=ModelConfig(),
        data=DataConfig(),
        training=TrainingConfig(),
        hardware=HardwareConfig(),
        logging=LoggingConfig()
    )