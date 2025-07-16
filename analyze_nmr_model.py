#!/usr/bin/env python3
"""
Analyze and visualize NMR to SMILES model performance
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from sklearn.metrics import confusion_matrix
import argparse
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NMRModelAnalyzer:
    """Analyze NMR to SMILES model performance"""
    
    def __init__(self, predictions_file: str):
        with open(predictions_file, 'r') as f:
            self.predictions = json.load(f)
        logger.info(f"Loaded {len(self.predictions)} predictions")
    
    def calculate_metrics(self):
        """Calculate various performance metrics"""
        metrics = {
            'total': len(self.predictions),
            'exact_matches': 0,
            'valid_smiles': 0,
            'tanimoto_similarities': [],
            'by_completeness': {'complete': {'total': 0, 'correct': 0}, 
                              'incomplete': {'total': 0, 'correct': 0}},
            'by_size': {}
        }
        
        for pred in self.predictions:
            if 'error' in pred:
                continue
            
            true_smiles = pred.get('true_smiles', '')
            pred_smiles = pred.get('predicted_smiles', '')
            
            # Exact match
            if true_smiles == pred_smiles:
                metrics['exact_matches'] += 1
            
            # Valid SMILES check
            pred_mol = Chem.MolFromSmiles(pred_smiles)
            if pred_mol is not None:
                metrics['valid_smiles'] += 1
                
                # Calculate Tanimoto similarity
                true_mol = Chem.MolFromSmiles(true_smiles)
                if true_mol is not None:
                    true_fp = FingerprintMols.FingerprintMol(true_mol)
                    pred_fp = FingerprintMols.FingerprintMol(pred_mol)
                    similarity = DataStructs.TanimotoSimilarity(true_fp, pred_fp)
                    metrics['tanimoto_similarities'].append(similarity)
            
            # By completeness
            is_complete = pred.get('is_complete', False)
            key = 'complete' if is_complete else 'incomplete'
            metrics['by_completeness'][key]['total'] += 1
            if true_smiles == pred_smiles:
                metrics['by_completeness'][key]['correct'] += 1
            
            # By molecular size
            n_atoms = pred.get('h_atoms', 0) + pred.get('c_atoms', 0)
            size_bin = self._get_size_bin(n_atoms)
            if size_bin not in metrics['by_size']:
                metrics['by_size'][size_bin] = {'total': 0, 'correct': 0}
            metrics['by_size'][size_bin]['total'] += 1
            if true_smiles == pred_smiles:
                metrics['by_size'][size_bin]['correct'] += 1
        
        return metrics
    
    def _get_size_bin(self, n_atoms):
        """Categorize molecules by size"""
        if n_atoms < 20:
            return 'small (<20)'
        elif n_atoms < 40:
            return 'medium (20-40)'
        elif n_atoms < 60:
            return 'large (40-60)'
        else:
            return 'very large (>60)'
    
    def plot_performance(self, metrics, output_dir='analysis_plots'):
        """Create performance visualization plots"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1. Overall accuracy pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        correct = metrics['exact_matches']
        incorrect = metrics['total'] - correct
        ax.pie([correct, incorrect], labels=['Correct', 'Incorrect'], 
               autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
        ax.set_title('Overall Exact Match Accuracy')
        plt.savefig(f'{output_dir}/overall_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Tanimoto similarity distribution
        if metrics['tanimoto_similarities']:
            fig, ax = plt.subplots(figsize=(10, 6))
            similarities = metrics['tanimoto_similarities']
            ax.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(similarities), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(similarities):.3f}')
            ax.set_xlabel('Tanimoto Similarity')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Tanimoto Similarities')
            ax.legend()
            plt.savefig(f'{output_dir}/tanimoto_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Performance by completeness
        fig, ax = plt.subplots(figsize=(8, 6))
        completeness_data = []
        for key, data in metrics['by_completeness'].items():
            if data['total'] > 0:
                accuracy = data['correct'] / data['total'] * 100
                completeness_data.append({
                    'Type': key.capitalize(),
                    'Accuracy': accuracy,
                    'Count': data['total']
                })
        
        if completeness_data:
            df = pd.DataFrame(completeness_data)
            bars = ax.bar(df['Type'], df['Accuracy'])
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Performance by Data Completeness')
            ax.set_ylim(0, 105)
            
            # Add count labels
            for i, (bar, count) in enumerate(zip(bars, df['Count'])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'n={count}', ha='center', va='bottom')
            
            plt.savefig(f'{output_dir}/performance_by_completeness.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Performance by molecular size
        fig, ax = plt.subplots(figsize=(10, 6))
        size_order = ['small (<20)', 'medium (20-40)', 'large (40-60)', 'very large (>60)']
        size_data = []
        
        for size in size_order:
            if size in metrics['by_size']:
                data = metrics['by_size'][size]
                if data['total'] > 0:
                    accuracy = data['correct'] / data['total'] * 100
                    size_data.append({
                        'Size': size,
                        'Accuracy': accuracy,
                        'Count': data['total']
                    })
        
        if size_data:
            df = pd.DataFrame(size_data)
            bars = ax.bar(df['Size'], df['Accuracy'])
            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Performance by Molecular Size')
            ax.set_ylim(0, 105)
            ax.tick_params(axis='x', rotation=45)
            
            # Add count labels
            for bar, count in zip(bars, df['Count']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'n={count}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/performance_by_size.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Plots saved to {output_dir}/")
    
    def analyze_errors(self, n_examples=10):
        """Analyze prediction errors"""
        errors = []
        
        for pred in self.predictions:
            if 'error' in pred:
                continue
            
            true_smiles = pred.get('true_smiles', '')
            pred_smiles = pred.get('predicted_smiles', '')
            
            if true_smiles != pred_smiles:
                # Calculate similarity for errors
                true_mol = Chem.MolFromSmiles(true_smiles)
                pred_mol = Chem.MolFromSmiles(pred_smiles)
                
                similarity = 0.0
                if true_mol and pred_mol:
                    true_fp = FingerprintMols.FingerprintMol(true_mol)
                    pred_fp = FingerprintMols.FingerprintMol(pred_mol)
                    similarity = DataStructs.TanimotoSimilarity(true_fp, pred_fp)
                
                errors.append({
                    'id': pred.get('id', 'unknown'),
                    'true_smiles': true_smiles,
                    'pred_smiles': pred_smiles,
                    'similarity': similarity,
                    'h_atoms': pred.get('h_atoms', 0),
                    'c_atoms': pred.get('c_atoms', 0),
                    'h_peaks': pred.get('h_peaks', 0),
                    'c_peaks': pred.get('c_peaks', 0),
                    'is_complete': pred.get('is_complete', False)
                })
        
        # Sort by similarity (worst predictions first)
        errors.sort(key=lambda x: x['similarity'])
        
        # Print error analysis
        print("\n" + "="*80)
        print("ERROR ANALYSIS")
        print("="*80)
        print(f"Total errors: {len(errors)}")
        
        if errors:
            print(f"\nWorst {min(n_examples, len(errors))} predictions:")
            print("-"*80)
            
            for i, error in enumerate(errors[:n_examples]):
                print(f"\n{i+1}. ID: {error['id']}")
                print(f"   True SMILES: {error['true_smiles']}")
                print(f"   Pred SMILES: {error['pred_smiles']}")
                print(f"   Similarity: {error['similarity']:.3f}")
                print(f"   Atoms: H={error['h_atoms']}, C={error['c_atoms']}")
                print(f"   Peaks: H={error['h_peaks']}, C={error['c_peaks']}")
                print(f"   Complete data: {error['is_complete']}")
        
        return errors
    
    def generate_report(self, output_file='analysis_report.txt'):
        """Generate comprehensive analysis report"""
        metrics = self.calculate_metrics()
        
        with open(output_file, 'w') as f:
            f.write("NMR TO SMILES MODEL ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write("OVERALL PERFORMANCE\n")
            f.write("-"*20 + "\n")
            f.write(f"Total predictions: {metrics['total']}\n")
            f.write(f"Exact matches: {metrics['exact_matches']} ({metrics['exact_matches']/metrics['total']*100:.1f}%)\n")
            f.write(f"Valid SMILES generated: {metrics['valid_smiles']} ({metrics['valid_smiles']/metrics['total']*100:.1f}%)\n")
            
            if metrics['tanimoto_similarities']:
                f.write(f"Average Tanimoto similarity: {np.mean(metrics['tanimoto_similarities']):.3f}\n")
                f.write(f"Median Tanimoto similarity: {np.median(metrics['tanimoto_similarities']):.3f}\n")
            
            f.write("\nPERFORMANCE BY DATA COMPLETENESS\n")
            f.write("-"*30 + "\n")
            for key, data in metrics['by_completeness'].items():
                if data['total'] > 0:
                    accuracy = data['correct'] / data['total'] * 100
                    f.write(f"{key.capitalize()}: {data['correct']}/{data['total']} ({accuracy:.1f}%)\n")
            
            f.write("\nPERFORMANCE BY MOLECULAR SIZE\n")
            f.write("-"*30 + "\n")
            size_order = ['small (<20)', 'medium (20-40)', 'large (40-60)', 'very large (>60)']
            for size in size_order:
                if size in metrics['by_size']:
                    data = metrics['by_size'][size]
                    if data['total'] > 0:
                        accuracy = data['correct'] / data['total'] * 100
                        f.write(f"{size}: {data['correct']}/{data['total']} ({accuracy:.1f}%)\n")
        
        logger.info(f"Report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze NMR to SMILES model performance')
    parser.add_argument('predictions', help='JSON file with predictions')
    parser.add_argument('--output-dir', default='analysis_results',
                       help='Output directory for plots and reports')
    parser.add_argument('--n-errors', type=int, default=10,
                       help='Number of error examples to show')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = NMRModelAnalyzer(args.predictions)
    
    # Calculate metrics
    metrics = analyzer.calculate_metrics()
    
    # Generate plots
    analyzer.plot_performance(metrics, str(output_dir / 'plots'))
    
    # Analyze errors
    analyzer.analyze_errors(n_examples=args.n_errors)
    
    # Generate report
    analyzer.generate_report(str(output_dir / 'analysis_report.txt'))
    
    logger.info(f"Analysis complete! Results saved to {output_dir}/")

if __name__ == "__main__":
    main()