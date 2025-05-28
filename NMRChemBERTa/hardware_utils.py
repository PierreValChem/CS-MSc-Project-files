"""
Hardware optimization utilities for NMR-ChemBERTa
"""
import os
import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging
import psutil
import platform

logger = logging.getLogger(__name__)


def get_optimal_device() -> str:
    """Automatically detect the best available device"""
    if torch.cuda.is_available():
        # Check for multiple GPUs
        gpu_count = torch.cuda.device_count()
        logger.info(f"CUDA available with {gpu_count} GPU(s)")
        
        # Log GPU details
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        return 'cuda'
    
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon) available")
        return 'mps'
    
    else:
        cpu_count = psutil.cpu_count()
        ram_gb = psutil.virtual_memory().total / 1024**3
        logger.info(f"Using CPU: {cpu_count} cores, {ram_gb:.1f} GB RAM")
        return 'cpu'


def setup_device(device_str: str = 'auto') -> torch.device:
    """Setup and return the appropriate device"""
    if device_str == 'auto':
        device_str = get_optimal_device()
    
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")
    
    # Set CUDA optimizations if using GPU
    if device.type == 'cuda':
        # Enable cuDNN benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True
        # Enable cuDNN determinism for reproducibility (may reduce performance)
        # torch.backends.cudnn.deterministic = True
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    return device


def optimize_model_for_hardware(model: nn.Module, 
                               device: torch.device,
                               compile_model: bool = True,
                               channels_last: bool = True) -> nn.Module:
    """Apply hardware-specific optimizations to the model"""
    
    # Move model to device
    model = model.to(device)
    
    # Apply memory format optimization
    if channels_last and device.type in ['cuda', 'cpu']:
        try:
            # Note: channels_last is primarily for conv nets, 
            # but can sometimes help with transformer models too
            model = model.to(memory_format=torch.channels_last)
            logger.info("Applied channels_last memory format")
        except Exception as e:
            logger.warning(f"Could not apply channels_last: {e}")
    
    # Apply PyTorch 2.0 compilation
    if compile_model and hasattr(torch, 'compile'):
        try:
            # Different compilation modes for different hardware
            if device.type == 'cuda':
                model = torch.compile(model, mode='max-autotune')
                logger.info("Applied torch.compile with max-autotune mode")
            else:
                model = torch.compile(model, mode='default')
                logger.info("Applied torch.compile with default mode")
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")
    
    return model


def get_optimal_batch_size(model: nn.Module, 
                          device: torch.device,
                          sample_input: dict,
                          max_batch_size: int = 64) -> int:
    """Automatically find optimal batch size"""
    if device.type == 'cpu':
        return min(8, max_batch_size)  # Conservative for CPU
    
    # GPU memory estimation
    model.eval()
    optimal_batch_size = 1
    
    with torch.no_grad():
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            if batch_size > max_batch_size:
                break
                
            try:
                # Create batch by repeating sample
                batch_input = {}
                for key, value in sample_input.items():
                    if isinstance(value, torch.Tensor):
                        batch_input[key] = value.unsqueeze(0).repeat(batch_size, *([1] * (value.dim())))
                    elif isinstance(value, dict):
                        batch_input[key] = {}
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, torch.Tensor):
                                batch_input[key][subkey] = subvalue.unsqueeze(0).repeat(
                                    batch_size, *([1] * (subvalue.dim()))
                                )
                
                # Test forward pass
                _ = model(**batch_input)
                optimal_batch_size = batch_size
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    break
                else:
                    raise e
    
    logger.info(f"Optimal batch size found: {optimal_batch_size}")
    return optimal_batch_size


def setup_mixed_precision() -> Tuple[torch.cuda.amp.GradScaler, bool]:
    """Setup mixed precision training"""
    use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    if use_amp:
        logger.info("Mixed precision training enabled (AMP)")
    else:
        logger.info("Mixed precision training disabled")
    
    return scaler, use_amp


def get_num_workers(batch_size: int) -> int:
    """Get optimal number of workers for data loading"""
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores
    
    # Rule of thumb: 2-4 workers per GPU, but not more than CPU cores
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        optimal_workers = min(cpu_count, num_gpus * 4, batch_size)
    else:
        optimal_workers = min(cpu_count, 8, batch_size)
    
    return max(0, optimal_workers)


def setup_distributed_training():
    """Setup for distributed training (multi-GPU)"""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return False, 0, 0
    
    # Check if running in distributed environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # Initialize process group
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(rank)
        
        logger.info(f"Distributed training: rank {rank}/{world_size}")
        return True, rank, world_size
    
    return False, 0, 0


def optimize_memory_usage():
    """Apply general memory optimizations"""
    import gc
    
    # Clear Python garbage collector
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Set memory fraction for CUDA (optional, uncomment if needed)
    # if torch.cuda.is_available():
    #     torch.cuda.set_per_process_memory_fraction(0.9)


def log_hardware_info():
    """Log detailed hardware information"""
    logger.info("=== Hardware Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"CPU: {platform.processor()}")
    logger.info(f"CPU cores: {psutil.cpu_count()} logical, {psutil.cpu_count(logical=False)} physical")
    logger.info(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            logger.info(f"GPU {i}: {props.name} ({memory_gb:.1f} GB, compute {props.major}.{props.minor})")
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info("==============================")


# Hardware-specific optimizations
class HardwareOptimizer:
    """Main class for handling hardware optimizations"""
    
    def __init__(self, config):
        self.config = config
        self.device = None
        self.scaler = None
        self.use_amp = False
    
    def setup(self):
        """Setup all hardware optimizations"""
        log_hardware_info()
        
        # Setup device
        self.device = setup_device(self.config.hardware.device)
        
        # Setup mixed precision
        if self.config.hardware.mixed_precision:
            self.scaler, self.use_amp = setup_mixed_precision()
        
        # Memory optimization
        optimize_memory_usage()
        
        return self.device, self.scaler, self.use_amp
    
    def optimize_model(self, model):
        """Apply model optimizations"""
        return optimize_model_for_hardware(
            model, 
            self.device,
            self.config.hardware.compile_model,
            self.config.hardware.channels_last
        )