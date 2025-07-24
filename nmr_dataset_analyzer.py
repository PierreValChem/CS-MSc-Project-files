#!/usr/bin/env python3
"""
NMR Dataset Visual Analysis Script
Generates comprehensive annotated graphs for dataset insights
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style for professional graphs
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

class NMRDatasetVisualizer:
    """Generate annotated visualizations for NMR dataset"""
    
    def __init__(self, data_dir: str, output_dir: str = 'dataset_visualizations'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data storage
        self.data = []
        self.complete_compounds = set()
        
        # Patterns for parsing
        self.h_pattern = re.compile(r'([\d.]+|NULL),\s*(\w+|null),.*,\s*([-\d]+)')
        self.c_pattern = re.compile(r'([\d.]+|NULL),\s*(\w+|null),\s*([-\d]+)')
    
    def parse_nmredata_file(self, filepath: str) -> Optional[Dict]:
        """Parse essential information from .nmredata file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            data = {}
            
            # Extract ID
            id_match = re.search(r'>  <NP_MRD_ID>\n(.+)', content)
            if id_match:
                data['id'] = id_match.group(1).strip()
            
            # Extract SMILES
            smiles_match = re.search(r'>  <Canonical_SMILES>\n(.+)', content)
            if smiles_match:
                data['smiles'] = smiles_match.group(1).strip()
                # Calculate basic molecular properties
                data['smiles_length'] = len(data['smiles'])
                data['num_rings'] = data['smiles'].count('1') + data['smiles'].count('2') + \
                                   data['smiles'].count('3') + data['smiles'].count('4') + \
                                   data['smiles'].count('5') + data['smiles'].count('6')
            
            # Extract atom counts
            h_atoms_match = re.search(r'H_atoms: (\d+)', content)
            c_atoms_match = re.search(r'C_atoms: (\d+)', content)
            if h_atoms_match:
                data['h_atoms'] = int(h_atoms_match.group(1))
            if c_atoms_match:
                data['c_atoms'] = int(c_atoms_match.group(1))
            
            # Extract completeness
            completeness_match = re.search(r'>  <DATA_COMPLETENESS>\n(.+)', content)
            if completeness_match:
                data['is_complete'] = 'COMPLETE' in completeness_match.group(1)
            
            # Parse 1H NMR data
            h_nmr_match = re.search(r'>  <NMREDATA_1D_1H>\n([\s\S]*?)(?=\n>|\n\$\$\$\$)', content)
            if h_nmr_match:
                h_peaks = self._parse_peaks(h_nmr_match.group(1), self.h_pattern)
                data['h_total_peaks'] = len(h_peaks)
                data['h_real_peaks'] = len([p for p in h_peaks if not p.get('is_padding', False)])
                data['h_padding_peaks'] = len([p for p in h_peaks if p.get('is_padding', False)])
                
                # Get multiplicities
                h_mults = [p['multiplicity'] for p in h_peaks if not p.get('is_padding', False)]
                data['h_multiplicities'] = Counter(h_mults)
            
            # Parse 13C NMR data
            c_nmr_match = re.search(r'>  <NMREDATA_1D_13C>\n([\s\S]*?)(?=\n>|\n\$\$\$\$)', content)
            if c_nmr_match:
                c_peaks = self._parse_peaks(c_nmr_match.group(1), self.c_pattern)
                data['c_total_peaks'] = len(c_peaks)
                data['c_real_peaks'] = len([p for p in c_peaks if not p.get('is_padding', False)])
                data['c_padding_peaks'] = len([p for p in c_peaks if p.get('is_padding', False)])
            
            return data
            
        except Exception as e:
            return None
    
    def _parse_peaks(self, peak_text: str, pattern) -> List[Dict]:
        """Parse peaks from NMR data"""
        peaks = []
        for line in peak_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
                
            match = pattern.match(line)
            if match:
                shift = match.group(1)
                multiplicity = match.group(2)
                atom_num = int(match.group(3))
                
                if shift == 'NULL' or shift == 'null' or atom_num == -1:
                    peaks.append({
                        'shift': None,
                        'multiplicity': 'null',
                        'atom_num': -1,
                        'is_padding': True
                    })
                else:
                    peaks.append({
                        'shift': float(shift) if shift else None,
                        'multiplicity': multiplicity if multiplicity else 'unknown',
                        'atom_num': atom_num,
                        'is_padding': False
                    })
        return peaks
    
    def load_complete_compounds(self):
        """Load list of complete compounds"""
        complete_list_path = self.data_dir / 'complete_data_compounds.txt'
        if complete_list_path.exists():
            with open(complete_list_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.strip() and not line.startswith('#'):
                        parts = line.strip().split(',')
                        if parts:
                            self.complete_compounds.add(parts[0].strip())
    
    def load_and_analyze(self):
        """Load dataset and create visualizations"""
        print("Loading NMR dataset...")
        
        # Load complete compounds list
        self.load_complete_compounds()
        
        # Find all .nmredata files
        nmredata_files = list(self.data_dir.glob('*.nmredata'))
        print(f"Found {len(nmredata_files)} .nmredata files")
        
        # Parse files
        for filepath in tqdm(nmredata_files, desc="Parsing files"):
            parsed_data = self.parse_nmredata_file(str(filepath))
            if parsed_data:
                # Add completeness flag
                if 'id' in parsed_data:
                    parsed_data['in_complete_list'] = parsed_data['id'] in self.complete_compounds
                self.data.append(parsed_data)
        
        print(f"Successfully parsed {len(self.data)} compounds")
        
        # Convert to DataFrame
        self.df = pd.DataFrame(self.data)
        
        # Generate all visualizations
        self.create_all_visualizations()
    
    def create_all_visualizations(self):
        """Generate all annotated graphs"""
        print("Generating visualizations...")
        
        # Print diagnostic information
        print(f"\nData Summary:")
        print(f"Total compounds: {len(self.df)}")
        if 'is_complete' in self.df:
            print(f"Completeness data available: {self.df['is_complete'].value_counts().to_dict()}")
        else:
            print("No completeness data found")
        
        if 'h_padding_peaks' in self.df:
            print(f"H padding range: {self.df['h_padding_peaks'].min():.0f} - {self.df['h_padding_peaks'].max():.0f}")
        if 'c_padding_peaks' in self.df:
            print(f"C padding range: {self.df['c_padding_peaks'].min():.0f} - {self.df['c_padding_peaks'].max():.0f}")
        
        print("\nGenerating plots...")
        
        # 1. Atom Distribution Analysis
        self.plot_atom_distributions()
        
        # 2. Padding Analysis
        self.plot_padding_analysis()
        
        # 3. Data Completeness Overview
        self.plot_completeness_overview()
        
        # 4. Peak vs Atom Correlations
        self.plot_peak_atom_correlations()
        
        # 5. Molecular Size Distribution
        self.plot_molecular_size_distribution()
        
        # 6. Spectral Quality Metrics
        self.plot_spectral_quality()
        
        # 7. Multiplicity Patterns
        self.plot_multiplicity_patterns()
        
        # 8. Comprehensive Dashboard
        self.create_summary_dashboard()
        
        print(f"All visualizations saved to {self.output_dir}")
    
    def plot_atom_distributions(self):
        """Plot H and C atom distributions with statistics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # H atoms distribution
        if 'h_atoms' in self.df:
            h_data = self.df['h_atoms'].dropna()
            ax1.hist(h_data, bins=50, color='dodgerblue', alpha=0.7, edgecolor='black')
            ax1.axvline(h_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {h_data.mean():.1f}')
            ax1.axvline(h_data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {h_data.median():.1f}')
            
            # Add text statistics
            stats_text = f'Total compounds: {len(h_data):,}\n'
            stats_text += f'Range: {h_data.min():.0f} - {h_data.max():.0f}\n'
            stats_text += f'Std Dev: {h_data.std():.1f}'
            ax1.text(0.7, 0.95, stats_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax1.set_xlabel('Number of H Atoms')
            ax1.set_ylabel('Count')
            ax1.set_title('Distribution of Hydrogen Atoms per Compound')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # C atoms distribution
        if 'c_atoms' in self.df:
            c_data = self.df['c_atoms'].dropna()
            ax2.hist(c_data, bins=50, color='forestgreen', alpha=0.7, edgecolor='black')
            ax2.axvline(c_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {c_data.mean():.1f}')
            ax2.axvline(c_data.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {c_data.median():.1f}')
            
            stats_text = f'Total compounds: {len(c_data):,}\n'
            stats_text += f'Range: {c_data.min():.0f} - {c_data.max():.0f}\n'
            stats_text += f'Std Dev: {c_data.std():.1f}'
            ax2.text(0.7, 0.95, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax2.set_xlabel('Number of C Atoms')
            ax2.set_ylabel('Count')
            ax2.set_title('Distribution of Carbon Atoms per Compound')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'atom_distributions.png', bbox_inches='tight')
        plt.close()
    
    def plot_padding_analysis(self):
        """Plot padding distribution and statistics"""
        fig = plt.figure(figsize=(16, 10))
        
        # Create subplots
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[2, :])
        
        # H padding distribution
        if 'h_padding_peaks' in self.df:
            h_padding = self.df['h_padding_peaks'].dropna()
            
            # Histogram
            ax1.hist(h_padding, bins=np.arange(0, min(50, h_padding.max()+2)), 
                    color='coral', alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Number of Padded H Peaks')
            ax1.set_ylabel('Count')
            ax1.set_title('H NMR Padding Distribution')
            
            # Add percentage annotations
            no_padding_h = (h_padding == 0).sum()
            total_h = len(h_padding)
            ax1.text(0.6, 0.9, f'No padding: {no_padding_h:,} ({no_padding_h/total_h*100:.1f}%)',
                    transform=ax1.transAxes, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # C padding distribution
        if 'c_padding_peaks' in self.df:
            c_padding = self.df['c_padding_peaks'].dropna()
            
            # Check if all values are 0
            if c_padding.max() == 0:
                ax2.text(0.5, 0.5, 'All C NMR data is complete\n(no padding required)', 
                        ha='center', va='center', transform=ax2.transAxes, 
                        fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                ax2.set_xlabel('Number of Padded C Peaks')
                ax2.set_ylabel('Count')
                ax2.set_title('C NMR Padding Distribution')
            else:
                # Histogram
                ax2.hist(c_padding, bins=np.arange(0, min(30, c_padding.max()+2)), 
                        color='lightblue', alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Number of Padded C Peaks')
                ax2.set_ylabel('Count')
                ax2.set_title('C NMR Padding Distribution')
                
                # Add percentage annotations
                no_padding_c = (c_padding == 0).sum()
                total_c = len(c_padding)
                ax2.text(0.6, 0.9, f'No padding: {no_padding_c:,} ({no_padding_c/total_c*100:.1f}%)',
                        transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Box plots for padding by completeness
        if 'is_complete' in self.df and 'h_padding_peaks' in self.df:
            # Get data for complete and incomplete compounds
            complete_mask = self.df['is_complete'] == True
            incomplete_mask = self.df['is_complete'] == False
            
            complete_data = self.df.loc[complete_mask, 'h_padding_peaks'].dropna()
            incomplete_data = self.df.loc[incomplete_mask, 'h_padding_peaks'].dropna()
            
            # Only plot if we have data for both categories
            plot_data = []
            plot_labels = []
            plot_colors = []
            
            if len(complete_data) > 0:
                plot_data.append(complete_data)
                plot_labels.append(f'Complete\n(n={len(complete_data):,})')
                plot_colors.append('lightgreen')
            
            if len(incomplete_data) > 0:
                plot_data.append(incomplete_data)
                plot_labels.append(f'Incomplete\n(n={len(incomplete_data):,})')
                plot_colors.append('lightcoral')
            
            if len(plot_data) > 0:
                bp1 = ax3.boxplot(plot_data, labels=plot_labels,
                                 patch_artist=True, notch=True)
                
                # Color the boxes
                for patch, color in zip(bp1['boxes'], plot_colors):
                    patch.set_facecolor(color)
                
                ax3.set_ylabel('H Padding Peaks')
                ax3.set_title('H Padding by Completeness Status')
                ax3.grid(True, alpha=0.3)
                
                # Add mean values
                for i, data in enumerate(plot_data):
                    if len(data) > 0:
                        ax3.text(i+1, data.max()*1.1, f'μ={data.mean():.1f}', 
                                ha='center', fontsize=10)
            else:
                ax3.text(0.5, 0.5, 'No data available for completeness comparison', 
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('H Padding by Completeness Status')
        else:
            ax3.text(0.5, 0.5, 'Completeness or padding data not available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('H Padding by Completeness Status')
        
        # Scatter plot: Total peaks vs Padding
        if 'h_total_peaks' in self.df and 'h_padding_peaks' in self.df:
            ax4.scatter(self.df['h_total_peaks'], self.df['h_padding_peaks'], 
                       alpha=0.5, s=20, c=self.df['h_atoms'] if 'h_atoms' in self.df else 'blue')
            ax4.set_xlabel('Total H Peaks')
            ax4.set_ylabel('Padded H Peaks')
            ax4.set_title('Total Peaks vs Padding')
            ax4.grid(True, alpha=0.3)
            
            # Add diagonal line
            max_val = max(self.df['h_total_peaks'].max(), self.df['h_padding_peaks'].max())
            ax4.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
            ax4.legend()
        
        # Summary statistics table
        if 'h_padding_peaks' in self.df and 'c_padding_peaks' in self.df:
            h_pad = self.df['h_padding_peaks']
            c_pad = self.df['c_padding_peaks']
            
            summary_data = [
                ['Metric', 'H NMR', 'C NMR'],
                ['Compounds with no padding', f'{(h_pad == 0).sum():,}', f'{(c_pad == 0).sum():,}'],
                ['Compounds with padding', f'{(h_pad > 0).sum():,}', f'{(c_pad > 0).sum():,}'],
                ['Average padding (all)', f'{h_pad.mean():.2f}', f'{c_pad.mean():.2f}'],
                ['Average padding (>0 only)', f'{h_pad[h_pad > 0].mean():.2f}', f'{c_pad[c_pad > 0].mean():.2f}'],
                ['Max padding', f'{h_pad.max():.0f}', f'{c_pad.max():.0f}'],
                ['Total padded peaks', f'{h_pad.sum():,}', f'{c_pad.sum():,}']
            ]
            
            ax5.axis('tight')
            ax5.axis('off')
            table = ax5.table(cellText=summary_data, cellLoc='center', loc='center',
                            colWidths=[0.4, 0.3, 0.3])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            
            # Style the header row
            for i in range(3):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.suptitle('NMR Padding Analysis', fontsize=16, y=0.98)
        plt.savefig(self.output_dir / 'padding_analysis.png', bbox_inches='tight')
        plt.close()
    
    def plot_completeness_overview(self):
        """Plot data completeness overview"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Pie chart of completeness
        if 'is_complete' in self.df:
            complete_counts = self.df['is_complete'].value_counts()
            
            # Handle case where we might have only one category
            if len(complete_counts) > 0:
                # Create a dictionary to ensure we have both categories
                counts_dict = {True: 0, False: 0}
                
                # Update with actual counts
                for key, value in complete_counts.items():
                    counts_dict[key] = value
                
                # Create lists for plotting
                values = []
                labels = []
                colors_to_use = []
                
                # Add False count first (incomplete)
                values.append(counts_dict[False])
                labels.append(f'Incomplete\n({counts_dict[False]:,} compounds)')
                colors_to_use.append('#ff9999')
                
                # Add True count (complete)
                values.append(counts_dict[True])
                labels.append(f'Complete\n({counts_dict[True]:,} compounds)')
                colors_to_use.append('#66b3ff')
                
                # Only plot non-zero values
                non_zero_values = []
                non_zero_labels = []
                non_zero_colors = []
                
                for val, lab, col in zip(values, labels, colors_to_use):
                    if val > 0:
                        non_zero_values.append(val)
                        non_zero_labels.append(lab)
                        non_zero_colors.append(col)
                
                if len(non_zero_values) > 0:
                    wedges, texts, autotexts = axes[0,0].pie(non_zero_values, 
                                                              labels=non_zero_labels,
                                                              colors=non_zero_colors,
                                                              autopct='%1.1f%%',
                                                              startangle=90)
                    axes[0,0].set_title('Dataset Completeness Distribution')
                else:
                    axes[0,0].text(0.5, 0.5, 'No data available', 
                                   ha='center', va='center', transform=axes[0,0].transAxes)
                    axes[0,0].set_title('Dataset Completeness Distribution')
            else:
                axes[0,0].text(0.5, 0.5, 'No completeness data available', 
                               ha='center', va='center', transform=axes[0,0].transAxes)
                axes[0,0].set_title('Dataset Completeness Distribution')
        else:
            axes[0,0].text(0.5, 0.5, 'Completeness field not found in data', 
                           ha='center', va='center', transform=axes[0,0].transAxes)
            axes[0,0].set_title('Dataset Completeness Distribution')
        
        # Bar chart comparing real vs padded peaks
        if 'h_real_peaks' in self.df and 'h_padding_peaks' in self.df:
            avg_data = {
                'Real H peaks': self.df['h_real_peaks'].mean(),
                'Padded H peaks': self.df['h_padding_peaks'].mean(),
                'Real C peaks': self.df['c_real_peaks'].mean() if 'c_real_peaks' in self.df else 0,
                'Padded C peaks': self.df['c_padding_peaks'].mean() if 'c_padding_peaks' in self.df else 0
            }
            
            bars = axes[0,1].bar(avg_data.keys(), avg_data.values(), 
                               color=['green', 'red', 'darkgreen', 'darkred'])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[0,1].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.1f}', ha='center', va='bottom')
            
            axes[0,1].set_ylabel('Average Count')
            axes[0,1].set_title('Average Real vs Padded Peaks')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Completeness score distribution
        if 'h_real_peaks' in self.df and 'h_total_peaks' in self.df:
            self.df['h_completeness'] = self.df['h_real_peaks'] / (self.df['h_total_peaks'] + 1e-6) * 100
            
            axes[1,0].hist(self.df['h_completeness'], bins=50, color='purple', alpha=0.7, edgecolor='black')
            axes[1,0].axvline(self.df['h_completeness'].mean(), color='red', linestyle='--', 
                            linewidth=2, label=f'Mean: {self.df["h_completeness"].mean():.1f}%')
            axes[1,0].set_xlabel('Completeness Score (%)')
            axes[1,0].set_ylabel('Count')
            axes[1,0].set_title('H NMR Completeness Score Distribution')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Stacked bar chart of compound categories
        if 'h_padding_peaks' in self.df and 'c_padding_peaks' in self.df:
            # Categorize compounds
            categories = []
            for _, row in self.df.iterrows():
                if pd.isna(row['h_padding_peaks']) or pd.isna(row['c_padding_peaks']):
                    categories.append('Unknown')
                elif row['h_padding_peaks'] == 0 and row['c_padding_peaks'] == 0:
                    categories.append('Perfect (no padding)')
                elif row['h_padding_peaks'] <= 5 and row['c_padding_peaks'] <= 2:
                    categories.append('Good (minimal padding)')
                elif row['h_padding_peaks'] <= 20 and row['c_padding_peaks'] <= 10:
                    categories.append('Moderate padding')
                else:
                    categories.append('Heavy padding')
            
            self.df['quality_category'] = categories
            category_counts = self.df['quality_category'].value_counts()
            
            # Define colors for each category
            color_map = {
                'Perfect (no padding)': 'darkgreen',
                'Good (minimal padding)': 'lightgreen',
                'Moderate padding': 'orange',
                'Heavy padding': 'red',
                'Unknown': 'gray'
            }
            
            # Get colors in the right order
            colors_cat = [color_map.get(cat, 'gray') for cat in category_counts.index]
            
            category_counts.plot(kind='bar', ax=axes[1,1], color=colors_cat)
            axes[1,1].set_title('Data Quality Categories')
            axes[1,1].set_xlabel('Category')
            axes[1,1].set_ylabel('Number of Compounds')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            # Add percentage labels
            total = len(self.df)
            for i, (idx, val) in enumerate(category_counts.items()):
                axes[1,1].text(i, val + total*0.01, f'{val/total*100:.1f}%', 
                             ha='center', va='bottom')
        else:
            axes[1,1].text(0.5, 0.5, 'Padding data not available', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Data Quality Categories')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'completeness_overview.png', bbox_inches='tight')
        plt.close()
    
    def plot_peak_atom_correlations(self):
        """Plot correlations between peaks and atoms"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
        
        # H peaks vs H atoms
        if 'h_atoms' in self.df and 'h_real_peaks' in self.df:
            # Scatter plot with density
            valid_mask = (self.df['h_atoms'] > 0) & (self.df['h_real_peaks'] > 0)
            x = self.df.loc[valid_mask, 'h_atoms']
            y = self.df.loc[valid_mask, 'h_real_peaks']
            
            # Create hexbin plot for density
            hb = ax1.hexbin(x, y, gridsize=30, cmap='YlOrRd', mincnt=1)
            cb = plt.colorbar(hb, ax=ax1)
            cb.set_label('Count')
            
            # Add ideal line (1:1)
            max_val = min(x.max(), y.max())
            ax1.plot([0, max_val], [0, max_val], 'b--', linewidth=2, label='Ideal (1:1)')
            
            # Add regression line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax1.plot(x.sort_values(), p(x.sort_values()), "r-", linewidth=2, 
                    label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
            
            ax1.set_xlabel('Number of H Atoms')
            ax1.set_ylabel('Number of H Peaks (Real)')
            ax1.set_title('H Peaks vs H Atoms Correlation')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # C peaks vs C atoms
        if 'c_atoms' in self.df and 'c_real_peaks' in self.df:
            valid_mask = (self.df['c_atoms'] > 0) & (self.df['c_real_peaks'] > 0)
            x = self.df.loc[valid_mask, 'c_atoms']
            y = self.df.loc[valid_mask, 'c_real_peaks']
            
            hb = ax2.hexbin(x, y, gridsize=30, cmap='YlGnBu', mincnt=1)
            cb = plt.colorbar(hb, ax=ax2)
            cb.set_label('Count')
            
            max_val = min(x.max(), y.max())
            ax2.plot([0, max_val], [0, max_val], 'b--', linewidth=2, label='Ideal (1:1)')
            
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax2.plot(x.sort_values(), p(x.sort_values()), "r-", linewidth=2, 
                    label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
            
            ax2.set_xlabel('Number of C Atoms')
            ax2.set_ylabel('Number of C Peaks (Real)')
            ax2.set_title('C Peaks vs C Atoms Correlation')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Peak/Atom ratio distributions
        if 'h_atoms' in self.df and 'h_real_peaks' in self.df:
            self.df['h_peak_ratio'] = self.df['h_real_peaks'] / (self.df['h_atoms'] + 1e-6)
            ratio_data = self.df.loc[self.df['h_peak_ratio'] <= 2, 'h_peak_ratio']
            
            ax3.hist(ratio_data, bins=50, color='indigo', alpha=0.7, edgecolor='black')
            ax3.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Ideal ratio (1.0)')
            ax3.axvline(ratio_data.mean(), color='green', linestyle='--', linewidth=2, 
                       label=f'Mean: {ratio_data.mean():.2f}')
            
            ax3.set_xlabel('H Peak/Atom Ratio')
            ax3.set_ylabel('Count')
            ax3.set_title('Distribution of H Peak/Atom Ratios')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add text box with statistics
            stats_text = f'Compounds with ratio = 1: {(abs(ratio_data - 1.0) < 0.01).sum():,}\n'
            stats_text += f'Compounds with ratio > 1: {(ratio_data > 1.0).sum():,}\n'
            stats_text += f'Compounds with ratio < 1: {(ratio_data < 1.0).sum():,}'
            ax3.text(0.65, 0.95, stats_text, transform=ax3.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # C Peak/Atom ratio
        if 'c_atoms' in self.df and 'c_real_peaks' in self.df:
            self.df['c_peak_ratio'] = self.df['c_real_peaks'] / (self.df['c_atoms'] + 1e-6)
            ratio_data = self.df.loc[self.df['c_peak_ratio'] <= 2, 'c_peak_ratio']
            
            ax4.hist(ratio_data, bins=50, color='teal', alpha=0.7, edgecolor='black')
            ax4.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Ideal ratio (1.0)')
            ax4.axvline(ratio_data.mean(), color='green', linestyle='--', linewidth=2, 
                       label=f'Mean: {ratio_data.mean():.2f}')
            
            ax4.set_xlabel('C Peak/Atom Ratio')
            ax4.set_ylabel('Count')
            ax4.set_title('Distribution of C Peak/Atom Ratios')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Add text box with statistics
            stats_text = f'Compounds with ratio = 1: {(abs(ratio_data - 1.0) < 0.01).sum():,}\n'
            stats_text += f'Compounds with ratio > 1: {(ratio_data > 1.0).sum():,}\n'
            stats_text += f'Compounds with ratio < 1: {(ratio_data < 1.0).sum():,}'
            ax4.text(0.65, 0.95, stats_text, transform=ax4.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'peak_atom_correlations.png', bbox_inches='tight')
        plt.close()
    
    def plot_molecular_size_distribution(self):
        """Plot molecular size metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Total atoms distribution
        if 'h_atoms' in self.df and 'c_atoms' in self.df:
            self.df['total_atoms'] = self.df['h_atoms'] + self.df['c_atoms']
            
            ax = axes[0,0]
            ax.hist(self.df['total_atoms'], bins=50, color='darkblue', alpha=0.7, edgecolor='black')
            ax.axvline(self.df['total_atoms'].mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {self.df["total_atoms"].mean():.1f}')
            ax.axvline(self.df['total_atoms'].median(), color='green', linestyle='--', 
                      linewidth=2, label=f'Median: {self.df["total_atoms"].median():.0f}')
            
            ax.set_xlabel('Total Atoms (H + C)')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Total Atom Count')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # H/C ratio distribution
        if 'h_atoms' in self.df and 'c_atoms' in self.df:
            self.df['h_c_ratio'] = self.df['h_atoms'] / (self.df['c_atoms'] + 1e-6)
            ratio_data = self.df.loc[self.df['h_c_ratio'] <= 10, 'h_c_ratio']
            
            ax = axes[0,1]
            ax.hist(ratio_data, bins=50, color='purple', alpha=0.7, edgecolor='black')
            ax.axvline(ratio_data.mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {ratio_data.mean():.2f}')
            
            # Add common ratios
            ax.axvline(2.0, color='orange', linestyle=':', linewidth=2, label='Alkanes (2.0)')
            ax.axvline(1.0, color='brown', linestyle=':', linewidth=2, label='Aromatics (1.0)')
            
            ax.set_xlabel('H/C Atomic Ratio')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of H/C Ratios')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Size categories
        if 'total_atoms' in self.df:
            # Define size categories
            size_bins = [0, 20, 40, 60, 100, 1000]
            size_labels = ['Very Small\n(<20)', 'Small\n(20-40)', 'Medium\n(40-60)', 
                          'Large\n(60-100)', 'Very Large\n(>100)']
            self.df['size_category'] = pd.cut(self.df['total_atoms'], bins=size_bins, labels=size_labels)
            
            size_counts = self.df['size_category'].value_counts()
            
            ax = axes[1,0]
            bars = ax.bar(size_counts.index, size_counts.values, 
                          color=['lightblue', 'skyblue', 'steelblue', 'darkblue', 'navy'])
            
            # Add count labels on bars
            for bar, count in zip(bars, size_counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count:,}\n({count/len(self.df)*100:.1f}%)', 
                       ha='center', va='bottom')
            
            ax.set_xlabel('Molecule Size Category')
            ax.set_ylabel('Number of Compounds')
            ax.set_title('Distribution by Molecular Size')
            ax.grid(True, alpha=0.3)
        
        # SMILES length as proxy for complexity
        if 'smiles_length' in self.df:
            ax = axes[1,1]
            smiles_data = self.df['smiles_length'].dropna()
            
            ax.hist(smiles_data, bins=50, color='darkgreen', alpha=0.7, edgecolor='black')
            ax.axvline(smiles_data.mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {smiles_data.mean():.1f}')
            
            # Add percentiles
            p25 = smiles_data.quantile(0.25)
            p75 = smiles_data.quantile(0.75)
            ax.axvspan(p25, p75, alpha=0.2, color='yellow', label='IQR (25%-75%)')
            
            ax.set_xlabel('SMILES String Length')
            ax.set_ylabel('Count')
            ax.set_title('SMILES Length Distribution (Complexity Proxy)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add text box
            stats_text = f'Min: {smiles_data.min():.0f}\n'
            stats_text += f'Max: {smiles_data.max():.0f}\n'
            stats_text += f'IQR: {p25:.0f} - {p75:.0f}'
            ax.text(0.7, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'molecular_size_distribution.png', bbox_inches='tight')
        plt.close()
    
    def plot_spectral_quality(self):
        """Plot spectral quality metrics"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Overall quality score
        ax1 = fig.add_subplot(gs[0, :2])
        if 'h_real_peaks' in self.df and 'h_total_peaks' in self.df:
            # Calculate quality scores
            self.df['h_quality_score'] = (self.df['h_real_peaks'] / (self.df['h_total_peaks'] + 1e-6)) * 100
            self.df['c_quality_score'] = (self.df['c_real_peaks'] / (self.df['c_total_peaks'] + 1e-6)) * 100 if 'c_real_peaks' in self.df else 0
            
            # Combined violin plot
            quality_data = []
            labels = []
            
            if 'h_quality_score' in self.df:
                quality_data.append(self.df['h_quality_score'].dropna())
                labels.append('H NMR')
            
            if 'c_quality_score' in self.df:
                quality_data.append(self.df['c_quality_score'].dropna())
                labels.append('C NMR')
            
            parts = ax1.violinplot(quality_data, positions=range(len(quality_data)), 
                                   showmeans=True, showmedians=True)
            
            # Color the violins
            colors = ['lightcoral', 'lightblue']
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels)
            ax1.set_ylabel('Quality Score (%)')
            ax1.set_title('NMR Spectral Quality Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Add statistics
            for i, data in enumerate(quality_data):
                ax1.text(i, 105, f'μ={data.mean():.1f}%\nσ={data.std():.1f}', 
                        ha='center', fontsize=10)
        
        # Quality score by completeness
        ax2 = fig.add_subplot(gs[0, 2])
        if 'is_complete' in self.df and 'h_quality_score' in self.df:
            complete_quality = self.df[self.df['is_complete'] == True]['h_quality_score'].dropna()
            incomplete_quality = self.df[self.df['is_complete'] == False]['h_quality_score'].dropna()
            
            bp = ax2.boxplot([complete_quality, incomplete_quality], 
                            labels=['Complete', 'Incomplete'],
                            patch_artist=True, notch=True)
            
            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][1].set_facecolor('lightcoral')
            
            ax2.set_ylabel('H Quality Score (%)')
            ax2.set_title('Quality by Completeness')
            ax2.grid(True, alpha=0.3)
        
        # Heatmap of padding patterns
        ax3 = fig.add_subplot(gs[1, :])
        if 'h_atoms' in self.df and 'h_padding_peaks' in self.df:
            # Create 2D histogram
            h_atoms_bins = np.linspace(0, min(50, self.df['h_atoms'].max()), 25)
            h_padding_bins = np.linspace(0, min(50, self.df['h_padding_peaks'].max()), 25)
            
            H, xedges, yedges = np.histogram2d(self.df['h_atoms'], self.df['h_padding_peaks'], 
                                               bins=[h_atoms_bins, h_padding_bins])
            
            im = ax3.imshow(H.T, origin='lower', aspect='auto', cmap='YlOrRd',
                           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            
            ax3.set_xlabel('Number of H Atoms')
            ax3.set_ylabel('Number of Padded H Peaks')
            ax3.set_title('Heatmap: H Atoms vs Padding Requirements')
            
            cb = plt.colorbar(im, ax=ax3)
            cb.set_label('Number of Compounds')
        
        # Padding categories pie chart
        ax4 = fig.add_subplot(gs[2, 0])
        if 'h_padding_peaks' in self.df:
            # Categorize padding
            padding_categories = []
            for val in self.df['h_padding_peaks']:
                if val == 0:
                    padding_categories.append('No padding')
                elif val <= 5:
                    padding_categories.append('Low (1-5)')
                elif val <= 15:
                    padding_categories.append('Medium (6-15)')
                else:
                    padding_categories.append('High (>15)')
            
            category_counts = pd.Series(padding_categories).value_counts()
            
            colors = ['darkgreen', 'lightgreen', 'orange', 'red']
            wedges, texts, autotexts = ax4.pie(category_counts.values, 
                                               labels=category_counts.index,
                                               colors=colors,
                                               autopct='%1.1f%%',
                                               startangle=90)
            
            ax4.set_title('H NMR Padding Categories')
        
        # Perfect vs imperfect data
        ax5 = fig.add_subplot(gs[2, 1])
        if 'h_padding_peaks' in self.df and 'c_padding_peaks' in self.df:
            perfect_count = ((self.df['h_padding_peaks'] == 0) & (self.df['c_padding_peaks'] == 0)).sum()
            imperfect_count = len(self.df) - perfect_count
            
            counts = [perfect_count, imperfect_count]
            labels = ['Perfect\n(no padding)', 'Imperfect\n(has padding)']
            colors = ['gold', 'lightgray']
            
            bars = ax5.bar(labels, counts, color=colors, edgecolor='black')
            
            # Add percentage labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count:,}\n({count/len(self.df)*100:.1f}%)', 
                        ha='center', va='bottom')
            
            ax5.set_ylabel('Number of Compounds')
            ax5.set_title('Data Perfection Status')
            ax5.grid(True, alpha=0.3)
        
        # Summary statistics
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        if 'h_quality_score' in self.df and 'c_quality_score' in self.df:
            summary_text = "Quality Summary\n" + "="*20 + "\n\n"
            summary_text += f"H NMR Avg Quality: {self.df['h_quality_score'].mean():.1f}%\n"
            summary_text += f"C NMR Avg Quality: {self.df['c_quality_score'].mean():.1f}%\n\n"
            summary_text += f"Perfect compounds: {((self.df['h_padding_peaks'] == 0) & (self.df['c_padding_peaks'] == 0)).sum():,}\n"
            summary_text += f"Total compounds: {len(self.df):,}\n\n"
            summary_text += f"Avg H padding: {self.df['h_padding_peaks'].mean():.2f}\n"
            summary_text += f"Avg C padding: {self.df['c_padding_peaks'].mean():.2f}"
            
            ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.suptitle('Spectral Quality Analysis', fontsize=16)
        plt.savefig(self.output_dir / 'spectral_quality_analysis.png', bbox_inches='tight')
        plt.close()
    
    def plot_multiplicity_patterns(self):
        """Plot multiplicity pattern analysis"""
        # Collect all multiplicities
        h_multiplicities = []
        c_multiplicities = []
        
        for _, row in self.df.iterrows():
            if 'h_multiplicities' in row and isinstance(row['h_multiplicities'], dict):
                h_multiplicities.extend(row['h_multiplicities'].items())
        
        if not h_multiplicities:
            return
        
        # Aggregate multiplicities
        h_mult_total = Counter()
        for mult, count in h_multiplicities:
            h_mult_total[mult] += count
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top multiplicities bar chart
        top_h_mults = dict(h_mult_total.most_common(15))
        
        bars = ax1.bar(range(len(top_h_mults)), list(top_h_mults.values()), 
                       color='steelblue', edgecolor='black')
        ax1.set_xticks(range(len(top_h_mults)))
        ax1.set_xticklabels(list(top_h_mults.keys()), rotation=45, ha='right')
        ax1.set_xlabel('Multiplicity Pattern')
        ax1.set_ylabel('Total Count')
        ax1.set_title('Top 15 H NMR Multiplicity Patterns')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom')
        
        # Multiplicity complexity distribution
        mult_complexity = {
            's': 1, 'd': 2, 't': 3, 'q': 4, 'p': 5, 
            'm': 6, 'dd': 7, 'dt': 8, 'td': 9, 'dq': 10,
            'ddd': 11, 'dtd': 12, 'null': 0, 'unknown': 0
        }
        
        # Calculate average multiplicity complexity per compound
        compound_complexities = []
        for _, row in self.df.iterrows():
            if 'h_multiplicities' in row and isinstance(row['h_multiplicities'], dict):
                total_complexity = 0
                total_peaks = 0
                for mult, count in row['h_multiplicities'].items():
                    complexity = mult_complexity.get(mult.lower() if isinstance(mult, str) else '', 0)
                    total_complexity += complexity * count
                    total_peaks += count
                if total_peaks > 0:
                    compound_complexities.append(total_complexity / total_peaks)
        
        if compound_complexities:
            ax2.hist(compound_complexities, bins=30, color='coral', alpha=0.7, edgecolor='black')
            ax2.axvline(np.mean(compound_complexities), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(compound_complexities):.2f}')
            ax2.set_xlabel('Average Multiplicity Complexity Score')
            ax2.set_ylabel('Number of Compounds')
            ax2.set_title('Distribution of Spectral Complexity')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add interpretation
            ax2.text(0.6, 0.95, 'Complexity Scale:\n1=singlet, 2=doublet\n3=triplet, etc.', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'multiplicity_patterns.png', bbox_inches='tight')
        plt.close()
    
    def create_summary_dashboard(self):
        """Create a comprehensive summary dashboard"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('NMR Dataset Comprehensive Analysis Dashboard', fontsize=20, y=0.98)
        
        # Key metrics (top row)
        ax_metrics = fig.add_subplot(gs[0, :])
        ax_metrics.axis('off')
        
        # Calculate key metrics
        total_compounds = len(self.df)
        complete_compounds = self.df['is_complete'].sum() if 'is_complete' in self.df else 0
        perfect_compounds = ((self.df['h_padding_peaks'] == 0) & (self.df['c_padding_peaks'] == 0)).sum() if 'h_padding_peaks' in self.df else 0
        avg_h_quality = self.df['h_quality_score'].mean() if 'h_quality_score' in self.df else 0
        
        metrics_text = f"""
        Total Compounds: {total_compounds:,}    |    Complete Compounds: {complete_compounds:,} ({complete_compounds/total_compounds*100:.1f}%)    |    
        Perfect Data (no padding): {perfect_compounds:,} ({perfect_compounds/total_compounds*100:.1f}%)    |    Average H NMR Quality: {avg_h_quality:.1f}%
        """
        
        ax_metrics.text(0.5, 0.5, metrics_text, transform=ax_metrics.transAxes,
                       fontsize=14, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
        
        # Atom distribution comparison
        ax1 = fig.add_subplot(gs[1, 0])
        if 'h_atoms' in self.df and 'c_atoms' in self.df:
            data_to_plot = [self.df['h_atoms'].dropna(), self.df['c_atoms'].dropna()]
            bp = ax1.boxplot(data_to_plot, labels=['H atoms', 'C atoms'], 
                            patch_artist=True, notch=True)
            bp['boxes'][0].set_facecolor('lightcoral')
            bp['boxes'][1].set_facecolor('lightgreen')
            ax1.set_ylabel('Count')
            ax1.set_title('Atom Distribution')
            ax1.grid(True, alpha=0.3)
        
        # Padding distribution comparison
        ax2 = fig.add_subplot(gs[1, 1])
        if 'h_padding_peaks' in self.df and 'c_padding_peaks' in self.df:
            padding_data = {
                'No padding': [
                    (self.df['h_padding_peaks'] == 0).sum(),
                    (self.df['c_padding_peaks'] == 0).sum()
                ],
                'With padding': [
                    (self.df['h_padding_peaks'] > 0).sum(),
                    (self.df['c_padding_peaks'] > 0).sum()
                ]
            }
            
            x = np.arange(2)
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, padding_data['No padding'], width, 
                           label='No padding', color='darkgreen')
            bars2 = ax2.bar(x + width/2, padding_data['With padding'], width, 
                           label='With padding', color='darkred')
            
            ax2.set_xticks(x)
            ax2.set_xticklabels(['H NMR', 'C NMR'])
            ax2.set_ylabel('Number of Compounds')
            ax2.set_title('Padding Status Comparison')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Quality score distribution
        ax3 = fig.add_subplot(gs[1, 2:])
        if 'h_quality_score' in self.df and 'c_quality_score' in self.df:
            ax3.hist(self.df['h_quality_score'], bins=50, alpha=0.5, label='H NMR', 
                    color='blue', density=True)
            ax3.hist(self.df['c_quality_score'], bins=50, alpha=0.5, label='C NMR', 
                    color='green', density=True)
            ax3.set_xlabel('Quality Score (%)')
            ax3.set_ylabel('Density')
            ax3.set_title('Quality Score Distributions')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Peak-to-atom ratio scatter
        ax4 = fig.add_subplot(gs[2, :2])
        if all(col in self.df for col in ['h_atoms', 'h_real_peaks', 'h_padding_peaks']):
            scatter = ax4.scatter(self.df['h_atoms'], self.df['h_real_peaks'], 
                                 c=self.df['h_padding_peaks'], s=20, alpha=0.5, 
                                 cmap='RdYlGn_r')
            ax4.plot([0, self.df['h_atoms'].max()], [0, self.df['h_atoms'].max()], 
                    'k--', alpha=0.5, label='Ideal (1:1)')
            
            cb = plt.colorbar(scatter, ax=ax4)
            cb.set_label('Padding Peaks')
            
            ax4.set_xlabel('H Atoms')
            ax4.set_ylabel('H Peaks (Real)')
            ax4.set_title('H Peak-Atom Relationship (colored by padding)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Molecular size categories
        ax5 = fig.add_subplot(gs[2, 2:])
        if 'total_atoms' in self.df:
            size_categories = pd.cut(self.df['total_atoms'], 
                                   bins=[0, 20, 40, 60, 100, 1000],
                                   labels=['<20', '20-40', '40-60', '60-100', '>100'])
            size_counts = size_categories.value_counts().sort_index()
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(size_counts)))
            wedges, texts, autotexts = ax5.pie(size_counts.values, labels=size_counts.index,
                                               colors=colors, autopct='%1.1f%%',
                                               startangle=90)
            ax5.set_title('Molecular Size Distribution\n(by total atom count)')
        
        # Summary statistics table
        ax6 = fig.add_subplot(gs[3, :])
        ax6.axis('tight')
        ax6.axis('off')
        
        # Create summary statistics
        summary_data = []
        if 'h_atoms' in self.df:
            summary_data.append(['H atoms', f'{self.df["h_atoms"].mean():.1f}', 
                               f'{self.df["h_atoms"].std():.1f}', 
                               f'{self.df["h_atoms"].min():.0f}', 
                               f'{self.df["h_atoms"].max():.0f}'])
        if 'c_atoms' in self.df:
            summary_data.append(['C atoms', f'{self.df["c_atoms"].mean():.1f}', 
                               f'{self.df["c_atoms"].std():.1f}', 
                               f'{self.df["c_atoms"].min():.0f}', 
                               f'{self.df["c_atoms"].max():.0f}'])
        if 'h_real_peaks' in self.df:
            summary_data.append(['H real peaks', f'{self.df["h_real_peaks"].mean():.1f}', 
                               f'{self.df["h_real_peaks"].std():.1f}', 
                               f'{self.df["h_real_peaks"].min():.0f}', 
                               f'{self.df["h_real_peaks"].max():.0f}'])
        if 'h_padding_peaks' in self.df:
            summary_data.append(['H padding', f'{self.df["h_padding_peaks"].mean():.1f}', 
                               f'{self.df["h_padding_peaks"].std():.1f}', 
                               f'{self.df["h_padding_peaks"].min():.0f}', 
                               f'{self.df["h_padding_peaks"].max():.0f}'])
        
        if summary_data:
            headers = ['Metric', 'Mean', 'Std Dev', 'Min', 'Max']
            table = ax6.table(cellText=summary_data, colLabels=headers,
                            cellLoc='center', loc='center',
                            colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 1.5)
            
            # Style the header
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.savefig(self.output_dir / 'comprehensive_dashboard.png', bbox_inches='tight', dpi=300)
        plt.close()

    def generate_html_report(self):
        """Generate comprehensive HTML report with all visualizations and analysis"""
        from datetime import datetime
        
        # Calculate all statistics needed for the report
        stats = self._calculate_comprehensive_statistics()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>NMR Dataset Analysis Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 25px;
        }}
        .summary-box {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #3498db;
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .image-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .image-container img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .insight-box {{
            background-color: #e8f8f5;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .warning-box {{
            background-color: #ffeaa7;
            border-left: 4px solid #fdcb6e;
            padding: 15px;
            margin: 20px 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>NMR Dataset Comprehensive Analysis Report</h1>
        
        <div class="summary-box">
            <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Dataset Path:</strong> {self.data_dir}</p>
            <p><strong>Total Compounds Analyzed:</strong> {len(self.df):,}</p>
        </div>
        
        <h2>Executive Summary</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Total Compounds</div>
                <div class="metric-value">{stats['total_compounds']:,}</div>
            </div>
            <div class="metric-card" style="background-color: #27ae60;">
                <div class="metric-label">Complete Compounds</div>
                <div class="metric-value">{stats['complete_compounds']:,}</div>
                <div class="metric-label">({stats['complete_percentage']:.1f}%)</div>
            </div>
            <div class="metric-card" style="background-color: #f39c12;">
                <div class="metric-label">Perfect Data (No Padding)</div>
                <div class="metric-value">{stats['perfect_compounds']:,}</div>
                <div class="metric-label">({stats['perfect_percentage']:.1f}%)</div>
            </div>
            <div class="metric-card" style="background-color: #9b59b6;">
                <div class="metric-label">Average H Quality</div>
                <div class="metric-value">{stats['avg_h_quality']:.1f}%</div>
            </div>
        </div>
        
        <div class="insight-box">
            <h3>Key Insights</h3>
            <ul>
                <li>The dataset contains {stats['total_compounds']:,} NMR spectra, with {stats['complete_percentage']:.1f}% marked as complete.</li>
                <li>Only {stats['perfect_percentage']:.1f}% of compounds have perfect peak assignments (no padding required).</li>
                <li>On average, H NMR spectra require {stats['avg_h_padding']:.1f} padded peaks, while C NMR requires {stats['avg_c_padding']:.1f}.</li>
                <li>The average H NMR quality score is {stats['avg_h_quality']:.1f}%, indicating {self._interpret_quality(stats['avg_h_quality'])}.</li>
            </ul>
        </div>
        
        <h2>1. Atom Distribution Analysis</h2>
        <p>This analysis shows the distribution of hydrogen and carbon atoms across all compounds in the dataset.</p>
        
        <div class="image-container">
            <img src="atom_distributions.png" alt="Atom Distributions">
        </div>
        
        <table>
            <tr>
                <th>Metric</th>
                <th>H Atoms</th>
                <th>C Atoms</th>
            </tr>
            <tr>
                <td>Average</td>
                <td>{stats['avg_h_atoms']:.1f}</td>
                <td>{stats['avg_c_atoms']:.1f}</td>
            </tr>
            <tr>
                <td>Median</td>
                <td>{stats['median_h_atoms']:.0f}</td>
                <td>{stats['median_c_atoms']:.0f}</td>
            </tr>
            <tr>
                <td>Range</td>
                <td>{stats['min_h_atoms']:.0f} - {stats['max_h_atoms']:.0f}</td>
                <td>{stats['min_c_atoms']:.0f} - {stats['max_c_atoms']:.0f}</td>
            </tr>
        </table>
        
        <div class="insight-box">
            <p><strong>Interpretation:</strong> The H/C ratio of approximately {stats['avg_h_c_ratio']:.2f} suggests the dataset contains 
            {self._interpret_h_c_ratio(stats['avg_h_c_ratio'])}.</p>
        </div>
        
        <h2>2. Padding Analysis</h2>
        <p>Padding analysis reveals how many peaks were added to standardize the dataset dimensions.</p>
        
        <div class="image-container">
            <img src="padding_analysis.png" alt="Padding Analysis">
        </div>
        
        <div class="warning-box">
            <h3>Data Quality Alert</h3>
            <p>{stats['compounds_with_h_padding']:,} compounds ({stats['h_padding_percentage']:.1f}%) require H NMR padding, 
            indicating incomplete peak assignments. This may impact model training quality.</p>
        </div>
        
        <h2>3. Data Completeness Overview</h2>
        <p>This section examines the overall completeness and quality of the spectral data.</p>
        
        <div class="image-container">
            <img src="completeness_overview.png" alt="Completeness Overview">
        </div>
        
        <table>
            <tr>
                <th>Quality Category</th>
                <th>Number of Compounds</th>
                <th>Percentage</th>
            </tr>
            <tr>
                <td>Perfect (no padding)</td>
                <td>{stats['perfect_compounds']:,}</td>
                <td>{stats['perfect_percentage']:.1f}%</td>
            </tr>
            <tr>
                <td>Good (minimal padding)</td>
                <td>{stats['good_compounds']:,}</td>
                <td>{stats['good_percentage']:.1f}%</td>
            </tr>
            <tr>
                <td>Moderate padding</td>
                <td>{stats['moderate_compounds']:,}</td>
                <td>{stats['moderate_percentage']:.1f}%</td>
            </tr>
            <tr>
                <td>Heavy padding</td>
                <td>{stats['heavy_compounds']:,}</td>
                <td>{stats['heavy_percentage']:.1f}%</td>
            </tr>
        </table>
        
        <h2>4. Peak-Atom Correlations</h2>
        <p>Analysis of the relationship between the number of atoms and observed NMR peaks.</p>
        
        <div class="image-container">
            <img src="peak_atom_correlations.png" alt="Peak-Atom Correlations">
        </div>
        
        <div class="insight-box">
            <p><strong>Correlation Analysis:</strong> The H peak/atom ratio of {stats['avg_h_peak_ratio']:.2f} indicates that 
            {self._interpret_peak_ratio(stats['avg_h_peak_ratio'], 'H')}. 
            The C peak/atom ratio of {stats['avg_c_peak_ratio']:.2f} shows 
            {self._interpret_peak_ratio(stats['avg_c_peak_ratio'], 'C')}.</p>
        </div>
        
        <h2>5. Molecular Size Distribution</h2>
        <p>Distribution of molecular sizes based on atom counts and structural features.</p>
        
        <div class="image-container">
            <img src="molecular_size_distribution.png" alt="Molecular Size Distribution">
        </div>
        
        <h2>6. Spectral Quality Analysis</h2>
        <p>Comprehensive analysis of spectral data quality and completeness scores.</p>
        
        <div class="image-container">
            <img src="spectral_quality_analysis.png" alt="Spectral Quality Analysis">
        </div>
        
        <h2>7. Multiplicity Patterns</h2>
        <p>Analysis of NMR multiplicity patterns and spectral complexity.</p>
        
        <div class="image-container">
            <img src="multiplicity_patterns.png" alt="Multiplicity Patterns">
        </div>
        
        <h2>8. Comprehensive Dashboard</h2>
        <p>Summary dashboard providing an overview of all key metrics.</p>
        
        <div class="image-container">
            <img src="comprehensive_dashboard.png" alt="Comprehensive Dashboard">
        </div>
        
        <h2>Recommendations</h2>
        <div class="warning-box">
            <h3>For Machine Learning Applications</h3>
            <ol>
                <li><strong>Use Complete Data:</strong> Consider training models primarily on the {stats['complete_compounds']:,} complete compounds 
                to ensure highest quality predictions.</li>
                <li><strong>Data Augmentation:</strong> The {stats['perfect_compounds']:,} compounds with perfect assignments could be used 
                for data augmentation strategies.</li>
                <li><strong>Quality Filtering:</strong> Consider filtering out compounds with quality scores below {stats['quality_threshold']:.0f}% 
                for critical applications.</li>
                <li><strong>Padding Strategy:</strong> With an average of {stats['avg_h_padding']:.1f} H padding peaks, ensure your model 
                architecture can effectively handle or ignore padded values.</li>
            </ol>
        </div>
        
        <div class="footer">
            <p>Generated by NMR Dataset Analyzer | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Dataset: {self.data_dir}</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        report_path = self.output_dir / 'nmr_dataset_analysis_report.html'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {report_path}")
        
        # Also generate a markdown version for easy reading
        self._generate_markdown_report(stats)
    
    def _calculate_comprehensive_statistics(self):
        """Calculate all statistics needed for the report"""
        stats = {}
        
        # Basic counts
        stats['total_compounds'] = len(self.df)
        stats['complete_compounds'] = self.df['is_complete'].sum() if 'is_complete' in self.df else 0
        stats['complete_percentage'] = (stats['complete_compounds'] / stats['total_compounds'] * 100) if stats['total_compounds'] > 0 else 0
        
        # Perfect compounds (no padding)
        if 'h_padding_peaks' in self.df and 'c_padding_peaks' in self.df:
            stats['perfect_compounds'] = ((self.df['h_padding_peaks'] == 0) & (self.df['c_padding_peaks'] == 0)).sum()
            stats['perfect_percentage'] = stats['perfect_compounds'] / stats['total_compounds'] * 100
        else:
            stats['perfect_compounds'] = 0
            stats['perfect_percentage'] = 0
        
        # Atom statistics
        if 'h_atoms' in self.df:
            stats['avg_h_atoms'] = self.df['h_atoms'].mean()
            stats['median_h_atoms'] = self.df['h_atoms'].median()
            stats['min_h_atoms'] = self.df['h_atoms'].min()
            stats['max_h_atoms'] = self.df['h_atoms'].max()
        
        if 'c_atoms' in self.df:
            stats['avg_c_atoms'] = self.df['c_atoms'].mean()
            stats['median_c_atoms'] = self.df['c_atoms'].median()
            stats['min_c_atoms'] = self.df['c_atoms'].min()
            stats['max_c_atoms'] = self.df['c_atoms'].max()
        
        # H/C ratio
        if 'h_atoms' in self.df and 'c_atoms' in self.df:
            stats['avg_h_c_ratio'] = (self.df['h_atoms'] / (self.df['c_atoms'] + 1e-6)).mean()
        
        # Padding statistics
        if 'h_padding_peaks' in self.df:
            stats['avg_h_padding'] = self.df['h_padding_peaks'].mean()
            stats['compounds_with_h_padding'] = (self.df['h_padding_peaks'] > 0).sum()
            stats['h_padding_percentage'] = stats['compounds_with_h_padding'] / stats['total_compounds'] * 100
        
        if 'c_padding_peaks' in self.df:
            stats['avg_c_padding'] = self.df['c_padding_peaks'].mean()
        
        # Quality scores
        if 'h_quality_score' in self.df:
            stats['avg_h_quality'] = self.df['h_quality_score'].mean()
        else:
            stats['avg_h_quality'] = 0
        
        # Peak ratios
        if 'h_peak_ratio' in self.df:
            stats['avg_h_peak_ratio'] = self.df['h_peak_ratio'].mean()
        else:
            stats['avg_h_peak_ratio'] = 0
            
        if 'c_peak_ratio' in self.df:
            stats['avg_c_peak_ratio'] = self.df['c_peak_ratio'].mean()
        else:
            stats['avg_c_peak_ratio'] = 0
        
        # Quality categories
        if 'quality_category' in self.df:
            for category in ['Perfect (no padding)', 'Good (minimal padding)', 'Moderate padding', 'Heavy padding']:
                count = (self.df['quality_category'] == category).sum()
                stats[f"{category.split()[0].lower()}_compounds"] = count
                stats[f"{category.split()[0].lower()}_percentage"] = count / stats['total_compounds'] * 100
        else:
            for prefix in ['perfect', 'good', 'moderate', 'heavy']:
                stats[f"{prefix}_compounds"] = 0
                stats[f"{prefix}_percentage"] = 0
        
        stats['quality_threshold'] = 80  # Recommended quality threshold
        
        return stats
    
    def _interpret_quality(self, quality_score):
        """Interpret quality score"""
        if quality_score >= 90:
            return "excellent spectral quality with minimal missing assignments"
        elif quality_score >= 80:
            return "good spectral quality with some missing assignments"
        elif quality_score >= 70:
            return "moderate spectral quality requiring careful consideration for ML applications"
        else:
            return "significant data incompleteness that may impact model performance"
    
    def _interpret_h_c_ratio(self, ratio):
        """Interpret H/C ratio"""
        if ratio > 2.2:
            return "primarily saturated compounds (alkane-like)"
        elif ratio > 1.5:
            return "a mix of saturated and unsaturated compounds"
        elif ratio > 1.0:
            return "significant aromatic or unsaturated character"
        else:
            return "highly aromatic or polycyclic compounds"
    
    def _interpret_peak_ratio(self, ratio, nucleus):
        """Interpret peak/atom ratio"""
        if ratio > 0.95:
            return f"nearly complete {nucleus} NMR assignments"
        elif ratio > 0.8:
            return f"good {nucleus} NMR coverage with some missing assignments"
        elif ratio > 0.6:
            return f"moderate {nucleus} NMR coverage with significant gaps"
        else:
            return f"poor {nucleus} NMR coverage requiring substantial padding"
    
    def _generate_markdown_report(self, stats):
        """Generate a markdown version of the report"""
        md_content = f"""# NMR Dataset Analysis Report


**Dataset:** {self.data_dir}

## Executive Summary

- **Total Compounds:** {stats['total_compounds']:,}
- **Complete Compounds:** {stats['complete_compounds']:,} ({stats['complete_percentage']:.1f}%)
- **Perfect Data (No Padding):** {stats['perfect_compounds']:,} ({stats['perfect_percentage']:.1f}%)
- **Average H NMR Quality:** {stats['avg_h_quality']:.1f}%

## Key Findings

1. The dataset contains {stats['total_compounds']:,} NMR spectra with varying levels of completeness.
2. Only {stats['perfect_percentage']:.1f}% of compounds have complete peak assignments without padding.
3. Average padding requirements: H NMR = {stats['avg_h_padding']:.1f} peaks, C NMR = {stats['avg_c_padding']:.1f} peaks.
4. The average H/C atomic ratio is {stats['avg_h_c_ratio']:.2f}, indicating {self._interpret_h_c_ratio(stats['avg_h_c_ratio'])}.

## Data Quality Distribution

| Quality Category | Count | Percentage |
|-----------------|-------|------------|
| Perfect (no padding) | {stats['perfect_compounds']:,} | {stats['perfect_percentage']:.1f}% |
| Good (minimal padding) | {stats['good_compounds']:,} | {stats['good_percentage']:.1f}% |
| Moderate padding | {stats['moderate_compounds']:,} | {stats['moderate_percentage']:.1f}% |
| Heavy padding | {stats['heavy_compounds']:,} | {stats['heavy_percentage']:.1f}% |

## Recommendations for Machine Learning

1. **Training Data Selection:** Use the {stats['complete_compounds']:,} complete compounds for highest quality model training.
2. **Quality Filtering:** Consider excluding compounds with quality scores below {stats['quality_threshold']}%.
3. **Padding Handling:** Implement masking strategies to handle the average {stats['avg_h_padding']:.1f} padded H peaks per compound.
4. **Data Augmentation:** Leverage the {stats['perfect_compounds']:,} perfect compounds for augmentation strategies.

## Visualizations

All detailed visualizations can be found in the accompanying HTML report and image files.
"""
        
        # Save markdown report
        md_path = self.output_dir / 'nmr_dataset_analysis_report.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"Markdown report generated: {md_path}")

def main():
    # Configuration
    data_dir = 'C:\\Users\\pierr\\Desktop\\CS MSc Project files\\peaklist\\complete_compounds_only'  # Update this path
    
    # Create visualizer and run analysis
    visualizer = NMRDatasetVisualizer(data_dir)
    visualizer.load_and_analyze()
    
    # Generate HTML report
    visualizer.generate_html_report()
    
    print(f"\nAnalysis complete!")
    print(f"Total compounds analyzed: {len(visualizer.df):,}")
    print(f"Visualizations saved to: {visualizer.output_dir}")
    print(f"HTML report: {visualizer.output_dir}/nmr_dataset_analysis_report.html")
    print(f"Markdown report: {visualizer.output_dir}/nmr_dataset_analysis_report.md")

if __name__ == "__main__":
    main()