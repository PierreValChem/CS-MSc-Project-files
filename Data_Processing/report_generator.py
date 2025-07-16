"""
Report generation module for NMReDATA converter - Updated for H NMR validation
"""

import os
import logging
from utils import setup_logging

logger = setup_logging()


class ReportGenerator:
    """Generates comprehensive accuracy reports"""
    
    def __init__(self, metrics, successful, failed, output_directory):
        self.metrics = metrics
        self.successful = successful
        self.failed = failed
        self.output_directory = output_directory
    
    def generate_report(self):
        """Generate comprehensive accuracy report"""
        # Calculate percentages
        total = self.metrics['total_compounds']
        total_original = total + self.metrics['compounds_without_nmr'] + self.metrics.get('compounds_without_h_nmr', 0)
        
        # Pre-check statistics
        nmr_coverage_pct = (self.metrics['compounds_with_nmr'] / total_original * 100) if total_original > 0 else 0
        h_nmr_missing_pct = (self.metrics.get('compounds_without_h_nmr', 0) / total_original * 100) if total_original > 0 else 0
        
        # Processing statistics
        txt_found_pct = (self.metrics['txt_files_found'] / total * 100) if total > 0 else 0
        valid_smiles_pct = (self.metrics['valid_smiles'] / total * 100) if total > 0 else 0
        gen_3d_pct = (self.metrics['3d_generation_success'] / self.metrics['valid_smiles'] * 100) if self.metrics['valid_smiles'] > 0 else 0
        peaks_parsed_pct = (self.metrics['peaks_parsed'] / self.metrics['txt_files_found'] * 100) if self.metrics['txt_files_found'] > 0 else 0
        mapping_pct = (self.metrics['atom_mapping_success'] / self.metrics['peaks_parsed'] * 100) if self.metrics['peaks_parsed'] > 0 else 0
        overall_success_pct = (self.successful / total * 100) if total > 0 else 0
        
        # Completeness statistics
        h_nmr_complete_pct = 100.0 - (self.metrics.get('h_nmr_completeness_warnings', 0) / total * 100) if total > 0 else 0
        c_nmr_complete_pct = 100.0 - (self.metrics.get('c_nmr_completeness_warnings', 0) / total * 100) if total > 0 else 0
        
        # Calculate accuracy score (weighted average of key metrics)
        accuracy_score = (
            nmr_coverage_pct * 0.1 +  # 10% weight for NMR data coverage
            txt_found_pct * 0.1 +  # 10% weight for finding files
            valid_smiles_pct * 0.15 +  # 15% weight for valid SMILES
            gen_3d_pct * 0.2 +  # 20% weight for 3D generation
            peaks_parsed_pct * 0.1 +  # 10% weight for parsing peaks
            mapping_pct * 0.1 +  # 10% weight for atom mapping
            h_nmr_complete_pct * 0.15 +  # 15% weight for H NMR completeness
            overall_success_pct * 0.1  # 10% weight for overall success
        ) / 100
        
        # Generate report
        report = self._format_report(
            total, total_original, nmr_coverage_pct, h_nmr_missing_pct, txt_found_pct,
            valid_smiles_pct, gen_3d_pct, peaks_parsed_pct, mapping_pct,
            overall_success_pct, h_nmr_complete_pct, c_nmr_complete_pct, accuracy_score
        )
        
        # Print and save report
        print(report)
        
        # Save detailed report to file
        report_path = os.path.join(self.output_directory, "accuracy_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
            # Add detailed error log
            if self.metrics['processing_errors']:
                f.write("\n\nDETAILED ERROR LOG:\n")
                f.write("="*80 + "\n")
                for error in self.metrics['processing_errors']:
                    f.write(f"\nCompound ID: {error['id']}\n")
                    f.write(f"Error: {error['error']}\n")
                    f.write("-"*40 + "\n")
        
        logger.info(f"Accuracy report saved to: {report_path}")
        
        # Return accuracy score for programmatic use
        return accuracy_score
    
    def _format_report(self, total, total_original, nmr_coverage_pct, h_nmr_missing_pct,
                      txt_found_pct, valid_smiles_pct, gen_3d_pct, peaks_parsed_pct,
                      mapping_pct, overall_success_pct, h_nmr_complete_pct, c_nmr_complete_pct,
                      accuracy_score):
        """Format the accuracy report"""
        report = f"""
================================================================================
                         NMReDATA CONVERSION ACCURACY REPORT
================================================================================

PRE-CHECK SUMMARY:
-----------------
Total Compounds in CSV:         {total_original}
Compounds with NMR data:        {self.metrics['compounds_with_nmr']} ({nmr_coverage_pct:.1f}%)
Compounds without NMR data:     {self.metrics['compounds_without_nmr']} ({100-nmr_coverage_pct:.1f}%)
Compounds without H NMR data:   {self.metrics.get('compounds_without_h_nmr', 0)} ({h_nmr_missing_pct:.1f}%)

PROCESSING SUMMARY:
------------------
Total Compounds Processed:      {total} (only those with H NMR data)
Successfully Converted:         {self.successful} ({overall_success_pct:.1f}%)
Failed:                        {self.failed} ({self.failed/total*100:.1f}% if total > 0 else 0)

DETAILED METRICS:
----------------
File Matching:
  - TXT files found:           {self.metrics['txt_files_found']} ({txt_found_pct:.1f}%)
  - TXT files missing:         {self.metrics['txt_files_missing']} ({self.metrics['txt_files_missing']/total*100:.1f}% if total > 0 else 0)

Structure Processing:
  - Valid SMILES:              {self.metrics['valid_smiles']} ({valid_smiles_pct:.1f}%)
  - Invalid SMILES:            {self.metrics['invalid_smiles']} ({self.metrics['invalid_smiles']/total*100:.1f}% if total > 0 else 0)
  - 3D generation success:     {self.metrics['3d_generation_success']} ({gen_3d_pct:.1f}% of valid SMILES)
  - 3D generation failed:      {self.metrics['3d_generation_failed']}
  - Empty MOL blocks:          {self.metrics.get('empty_mol_blocks', 0)}

NMR Data Processing:
  - Peaklists parsed:          {self.metrics['peaks_parsed']} ({peaks_parsed_pct:.1f}% of found files)
  - Empty peaklists:           {self.metrics['empty_peaklists']}
  - Atom mapping success:      {self.metrics['atom_mapping_success']} ({mapping_pct:.1f}% of parsed peaks)
  - Atom mapping failed:       {self.metrics['atom_mapping_failed']}

NMR Completeness:
  - H NMR complete spectra:    {h_nmr_complete_pct:.1f}%
  - H NMR incomplete warnings: {self.metrics.get('h_nmr_completeness_warnings', 0)}
  - C NMR complete spectra:    {c_nmr_complete_pct:.1f}%
  - C NMR incomplete warnings: {self.metrics.get('c_nmr_completeness_warnings', 0)}

Output:
  - NMReDATA files created:    {self.metrics['nmredata_created']}

System Issues:
  - Memory issues:             {self.metrics.get('memory_issues', 0)}
  - Timeout issues:            {self.metrics.get('timeout_issues', 0)}

ACCURACY SCORE: {accuracy_score:.2%}
--------------
(Weighted average of all metrics)

QUALITY INDICATORS:
------------------
✓ Excellent (>95%): """ + ("✓" if accuracy_score > 0.95 else "✗") + f"""
✓ Good (85-95%):    """ + ("✓" if 0.85 <= accuracy_score <= 0.95 else "✗") + f"""
✓ Fair (70-85%):    """ + ("✓" if 0.70 <= accuracy_score < 0.85 else "✗") + f"""
✗ Poor (<70%):      """ + ("✓" if accuracy_score < 0.70 else "✗") + f"""

PROCESSING MODE:
---------------
✓ Consolidation: DISABLED (all peaks kept individually)
✓ H NMR Required: YES (compounds without H NMR are skipped)
✓ Completeness Check: ENABLED (warnings for incomplete spectra)
✓ Molecular Representations: ENHANCED (SMILES, InChI, InChIKey, etc.)

COMMON ISSUES:
-------------"""
        
        # Add common issues
        issues = []
        
        if self.metrics['compounds_without_nmr'] > 0:
            issues.append(f"- Missing NMR data: {self.metrics['compounds_without_nmr']} compounds had no NMR data files")
        
        if self.metrics.get('compounds_without_h_nmr', 0) > 0:
            issues.append(f"- Missing H NMR: {self.metrics.get('compounds_without_h_nmr', 0)} compounds had no H NMR data (skipped)")
        
        if self.metrics['invalid_smiles'] > 0:
            issues.append(f"- Invalid SMILES: {self.metrics['invalid_smiles']} compounds have unparseable structures")
        
        if self.metrics['3d_generation_failed'] > 0:
            issues.append(f"- 3D generation failures: {self.metrics['3d_generation_failed']} molecules couldn't be embedded with MMFF94s")
        
        if self.metrics.get('empty_mol_blocks', 0) > 0:
            issues.append(f"- Empty MOL blocks: {self.metrics.get('empty_mol_blocks', 0)} files were not created due to invalid structure data")
        
        if self.metrics['atom_mapping_failed'] > 0:
            issues.append(f"- Atom mapping issues: {self.metrics['atom_mapping_failed']} compounds have ambiguous NMR assignments")
        
        if self.metrics.get('h_nmr_completeness_warnings', 0) > 0:
            issues.append(f"- Incomplete H NMR: {self.metrics.get('h_nmr_completeness_warnings', 0)} compounds have incomplete H NMR spectra")
        
        if self.metrics.get('c_nmr_completeness_warnings', 0) > 0:
            issues.append(f"- Incomplete C NMR: {self.metrics.get('c_nmr_completeness_warnings', 0)} compounds have incomplete C NMR spectra")
        
        if issues:
            report += "\n" + "\n".join(issues)
        else:
            report += "\nNo significant issues detected."
        
        # Add error samples if any
        if self.metrics['processing_errors']:
            report += f"\n\nSAMPLE ERRORS (first 5):\n"
            for error in self.metrics['processing_errors'][:5]:
                report += f"- {error['id']}: {error['error'][:100]}...\n"
        
        report += """
================================================================================
"""
        
        return report