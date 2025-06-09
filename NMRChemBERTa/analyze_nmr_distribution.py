"""
Analyze NMR data distribution to find optimal normalization parameters
"""
import numpy as np
from nmr_dataset import NMRDataset
from config import Config
import matplotlib.pyplot as plt

def analyze_nmr_distribution(config):
    """Analyze the distribution of NMR values in the dataset"""
    
    # Load dataset
    dataset = NMRDataset(config, split='full')
    
    all_h_shifts = []
    all_c_shifts = []
    
    print("Analyzing NMR data distribution...")
    
    # Collect all NMR values
    for i in range(len(dataset)):
        sample = dataset.data[i]
        nmr_data = sample['nmr_data']
        
        # Collect H shifts
        if nmr_data['H']['shifts']:
            all_h_shifts.extend(nmr_data['H']['shifts'])
        
        # Collect C shifts  
        if nmr_data['C']['shifts']:
            all_c_shifts.extend(nmr_data['C']['shifts'])
    
    # Convert to numpy arrays
    all_h_shifts = np.array(all_h_shifts)
    all_c_shifts = np.array(all_c_shifts)
    
    # Calculate statistics
    h_stats = {
        'mean': np.mean(all_h_shifts),
        'std': np.std(all_h_shifts),
        'min': np.min(all_h_shifts),
        'max': np.max(all_h_shifts),
        'median': np.median(all_h_shifts),
        'percentiles': {
            '5': np.percentile(all_h_shifts, 5),
            '95': np.percentile(all_h_shifts, 95)
        }
    }
    
    c_stats = {
        'mean': np.mean(all_c_shifts),
        'std': np.std(all_c_shifts),
        'min': np.min(all_c_shifts),
        'max': np.max(all_c_shifts),
        'median': np.median(all_c_shifts),
        'percentiles': {
            '5': np.percentile(all_c_shifts, 5),
            '95': np.percentile(all_c_shifts, 95)
        }
    }
    
    print("\n¹H NMR Statistics:")
    print(f"  Mean: {h_stats['mean']:.2f} ppm")
    print(f"  Std: {h_stats['std']:.2f} ppm")
    print(f"  Range: [{h_stats['min']:.2f}, {h_stats['max']:.2f}] ppm")
    print(f"  5-95 percentile: [{h_stats['percentiles']['5']:.2f}, {h_stats['percentiles']['95']:.2f}] ppm")
    
    print("\n¹³C NMR Statistics:")
    print(f"  Mean: {c_stats['mean']:.2f} ppm")
    print(f"  Std: {c_stats['std']:.2f} ppm")
    print(f"  Range: [{c_stats['min']:.2f}, {c_stats['max']:.2f}] ppm")
    print(f"  5-95 percentile: [{c_stats['percentiles']['5']:.2f}, {c_stats['percentiles']['95']:.2f}] ppm")
    
    # Plot distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(all_h_shifts, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(h_stats['mean'], color='red', linestyle='--', label=f"Mean: {h_stats['mean']:.2f}")
    ax1.set_xlabel('Chemical Shift (ppm)')
    ax1.set_ylabel('Count')
    ax1.set_title('¹H NMR Distribution')
    ax1.legend()
    
    ax2.hist(all_c_shifts, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(c_stats['mean'], color='red', linestyle='--', label=f"Mean: {c_stats['mean']:.2f}")
    ax2.set_xlabel('Chemical Shift (ppm)')
    ax2.set_ylabel('Count')
    ax2.set_title('¹³C NMR Distribution')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('nmr_distribution.png', dpi=150)
    plt.close()
    
    print("\nDistribution plot saved as 'nmr_distribution.png'")
    
    # Suggest normalization parameters
    print("\n" + "="*50)
    print("RECOMMENDED NORMALIZATION PARAMETERS:")
    print("="*50)
    print(f"H NMR: mean={h_stats['mean']:.2f}, std={h_stats['std']:.2f}")
    print(f"C NMR: mean={c_stats['mean']:.2f}, std={c_stats['std']:.2f}")
    print("\nUpdate these values in nmr_dataset.py:")
    print(f"self.nmr_norm_params = {{")
    print(f"    'h_shift': {{'mean': {h_stats['mean']:.1f}, 'std': {h_stats['std']:.1f}}},")
    print(f"    'c_shift': {{'mean': {c_stats['mean']:.1f}, 'std': {c_stats['std']:.1f}}}")
    print(f"}}")
    
    return h_stats, c_stats

if __name__ == "__main__":
    config = Config.from_yaml('config_smiles_nmr.yaml')
    analyze_nmr_distribution(config)