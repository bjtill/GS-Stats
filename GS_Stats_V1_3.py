#!/usr/bin/env python3
"""
Call Rate Analysis Program

Analyzes genotype call rates by sample and marker, comparing groups defined by
age (Bin) and gender (Breed). Performs statistical tests and generates an HTML report.

Usage:
    python call_rate_analysis.py genotype_file.txt sample_age_bin_table.txt -o output_dir [-c threshold]

Author: Generated with Claude
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import warnings

# Suppress specific warnings that occur with small sample sizes
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Degrees of freedom.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*divide by zero.*')
warnings.filterwarnings('ignore', message='.*ConstantInput.*')
warnings.filterwarnings('ignore', message='.*constant.*')

# For post-hoc tests
import scikit_posthocs as sp


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze genotype call rates by group with statistical comparisons.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python call_rate_analysis.py genotypes.txt breed_bin.txt -o my_analysis
    python call_rate_analysis.py genotypes.txt breed_bin.txt -o my_analysis -c 50
        """
    )
    
    parser.add_argument('genotype_file', 
                        help='Raw genotype file (Marker, Sample, Genotype) - no header, tab/space delimited')
    parser.add_argument('breed_bin_table', 
                        help='Sample metadata file (Sample, Breed, Bin) - tab delimited with header')
    parser.add_argument('-o', '--output', required=True,
                        help='Name for output directory')
    parser.add_argument('-c', '--callrate', type=float, default=None,
                        help='Minimum sample call rate threshold (0-100). Samples below this are excluded.')
    parser.add_argument('--exclude-failed-markers', action='store_true',
                        help='Exclude markers that failed in ALL samples before calculating call rates.')
    parser.add_argument('--skip-marker-analysis', action='store_true',
                        help='Skip marker call rate analysis entirely (only perform sample analysis).')
    
    return parser.parse_args()


def load_genotype_data(filepath):
    """Load raw genotype data from file."""
    print(f"Loading genotype data from {filepath}...")
    
    # First, peek at the file to determine format
    with open(filepath, 'r') as f:
        first_lines = [f.readline() for _ in range(5)]
    
    print(f"  First line preview: {repr(first_lines[0][:100])}")
    
    # Determine delimiter
    first_line = first_lines[0]
    if '\t' in first_line:
        sep = '\t'
        sep_name = 'tab'
    elif ',' in first_line:
        sep = ','
        sep_name = 'comma'
    else:
        sep = r'\s+'
        sep_name = 'whitespace'
    
    print(f"  Detected delimiter: {sep_name}")
    
    # Check number of fields in first line
    if sep == r'\s+':
        n_fields = len(first_line.split())
    else:
        n_fields = len(first_line.split(sep))
    
    print(f"  Fields detected in first line: {n_fields}")
    
    if n_fields != 3:
        print(f"  WARNING: Expected 3 columns (Marker, Sample, Genotype), found {n_fields}")
        split_result = first_line.split() if sep == r'\s+' else first_line.split(sep)
        print(f"  First line split: {split_result}")
    
    # Load the data
    try:
        df = pd.read_csv(filepath, sep=sep, header=None, names=['Marker', 'Sample', 'Genotype'],
                         dtype=str, engine='python')
    except Exception as e:
        print(f"  Error with {sep_name} delimiter: {e}")
        print("  Trying alternative parsing...")
        df = pd.read_csv(filepath, sep=r'\s+', header=None, names=['Marker', 'Sample', 'Genotype'],
                         dtype=str, engine='python')
    
    # Check if we got the right number of columns
    if len(df.columns) != 3:
        raise ValueError(f"Expected 3 columns, got {len(df.columns)}. "
                        f"Please check file format (should be: Marker Sample Genotype)")
    
    # Check for parsing issues
    print(f"  Sample of parsed data:")
    print(f"    Row 0: Marker='{df.iloc[0]['Marker']}', Sample='{df.iloc[0]['Sample']}', Genotype='{df.iloc[0]['Genotype']}'")
    
    # Ensure all columns are strings and standardize genotype values
    df['Marker'] = df['Marker'].astype(str).str.strip()
    df['Sample'] = df['Sample'].astype(str).str.strip()
    df['Genotype'] = df['Genotype'].astype(str).str.upper().str.strip()
    
    print(f"  Loaded {len(df):,} genotype records")
    print(f"  Unique markers: {df['Marker'].nunique():,}")
    print(f"  Unique samples: {df['Sample'].nunique():,}")
    
    # Show genotype distribution
    geno_counts = df['Genotype'].value_counts()
    print(f"  Genotype distribution: {geno_counts.head(10).to_dict()}")
    
    return df


def load_breed_bin_table(filepath):
    """Load breed/bin metadata table."""
    print(f"Loading breed/bin table from {filepath}...")
    
    # First, peek at the file to determine format
    with open(filepath, 'r') as f:
        first_lines = [f.readline() for _ in range(5)]
    
    print(f"  First line preview: {repr(first_lines[0][:100])}")
    if len(first_lines) > 1:
        print(f"  Second line preview: {repr(first_lines[1][:100])}")
    
    # Determine delimiter
    first_line = first_lines[0]
    if '\t' in first_line:
        sep = '\t'
        sep_name = 'tab'
    elif ',' in first_line:
        sep = ','
        sep_name = 'comma'
    else:
        sep = r'\s+'
        sep_name = 'whitespace'
    
    print(f"  Detected delimiter: {sep_name}")
    
    # Check number of fields
    if sep == r'\s+':
        n_fields = len(first_line.split())
    else:
        n_fields = len(first_line.split(sep))
    
    print(f"  Fields detected in first line: {n_fields}")
    
    # Determine if there's a header
    first_field = first_line.split()[0] if first_line.split() else ''
    has_header = first_field.lower() in ['sample', 'samples', 'id', 'name', 'sampleid', 'sample_id']
    
    print(f"  Header detected: {has_header}")
    
    # Load the data
    try:
        if has_header:
            df = pd.read_csv(filepath, sep=sep, header=0, dtype=str, engine='python')
        else:
            df = pd.read_csv(filepath, sep=sep, header=None, dtype=str, engine='python')
    except Exception as e:
        print(f"  Error with {sep_name} delimiter: {e}")
        print("  Trying alternative parsing...")
        df = pd.read_csv(filepath, sep=r'\s+', header=None, dtype=str, engine='python')
    
    print(f"  Columns found: {list(df.columns)}")
    print(f"  Shape: {df.shape}")
    
    # Standardize column names
    if len(df.columns) == 3:
        df.columns = ['Sample', 'Breed', 'Bin']
    elif len(df.columns) > 3:
        print(f"  WARNING: Found {len(df.columns)} columns, expected 3. Using first 3 columns.")
        df = df.iloc[:, :3]
        df.columns = ['Sample', 'Breed', 'Bin']
    else:
        raise ValueError(f"Expected 3 columns (Sample, Breed, Bin), found {len(df.columns)}. "
                        f"Columns: {list(df.columns)}")
    
    # Clean up values
    df['Sample'] = df['Sample'].astype(str).str.strip()
    df['Breed'] = df['Breed'].astype(str).str.upper().str.strip()
    df['Bin'] = df['Bin'].astype(str).str.upper().str.strip()
    
    print(f"  Sample of parsed data:")
    print(f"    Row 0: Sample='{df.iloc[0]['Sample']}', Breed='{df.iloc[0]['Breed']}', Bin='{df.iloc[0]['Bin']}'")
    
    print(f"  Loaded {len(df)} sample annotations")
    print(f"  Breeds: {df['Breed'].value_counts().to_dict()}")
    print(f"  Bins: {df['Bin'].value_counts().to_dict()}")
    
    return df


def calculate_sample_call_rates(genotype_df, valid_genotypes={'AA', 'BB', 'AB'}):
    """Calculate call rate for each sample."""
    # Mark passing genotypes
    genotype_df['Pass'] = genotype_df['Genotype'].isin(valid_genotypes)
    
    # Group by sample
    sample_stats = genotype_df.groupby('Sample').agg(
        Total_Markers=('Marker', 'count'),
        Passing_Markers=('Pass', 'sum')
    ).reset_index()
    
    sample_stats['Call_Rate'] = (sample_stats['Passing_Markers'] / sample_stats['Total_Markers']) * 100
    
    return sample_stats


def calculate_marker_call_rates(genotype_df, sample_list=None, valid_genotypes={'AA', 'BB', 'AB'}):
    """Calculate call rate for each marker, optionally filtering to specific samples."""
    df = genotype_df.copy()
    
    if sample_list is not None:
        df = df[df['Sample'].isin(sample_list)]
    
    # Mark passing genotypes
    df['Pass'] = df['Genotype'].isin(valid_genotypes)
    
    # Group by marker
    marker_stats = df.groupby('Marker').agg(
        Total_Samples=('Sample', 'count'),
        Passing_Samples=('Pass', 'sum')
    ).reset_index()
    
    marker_stats['Call_Rate'] = (marker_stats['Passing_Samples'] / marker_stats['Total_Samples']) * 100
    
    return marker_stats


def filter_samples_by_callrate(sample_call_rates, threshold):
    """Filter out samples below call rate threshold."""
    passing = sample_call_rates[sample_call_rates['Call_Rate'] >= threshold]['Sample'].tolist()
    failing = sample_call_rates[sample_call_rates['Call_Rate'] < threshold].copy()
    
    return passing, failing


def exclude_failed_markers(genotype_df, valid_genotypes={'AA', 'BB', 'AB'}):
    """
    Identify and remove markers that failed in ALL samples.
    
    Returns:
        filtered_df: DataFrame with failed markers removed
        failed_markers: List of marker names that were excluded
    """
    print("\nIdentifying completely failed markers...")
    
    # Mark passing genotypes
    df = genotype_df.copy()
    df['Pass'] = df['Genotype'].isin(valid_genotypes)
    
    # Group by marker and check if ANY sample has a passing call
    marker_stats = df.groupby('Marker').agg(
        Total_Samples=('Sample', 'count'),
        Passing_Samples=('Pass', 'sum')
    ).reset_index()
    
    # Identify markers with zero passing samples
    failed_markers = marker_stats[marker_stats['Passing_Samples'] == 0]['Marker'].tolist()
    
    print(f"  Total markers: {genotype_df['Marker'].nunique():,}")
    print(f"  Completely failed markers: {len(failed_markers):,}")
    
    if failed_markers:
        print(f"  Removing failed markers from analysis...")
        filtered_df = genotype_df[~genotype_df['Marker'].isin(failed_markers)].copy()
        print(f"  Remaining markers: {filtered_df['Marker'].nunique():,}")
    else:
        print(f"  No completely failed markers found.")
        filtered_df = genotype_df.copy()
    
    return filtered_df, failed_markers


def run_statistical_tests(data_dict, alpha=0.05):
    """
    Run omnibus and post-hoc statistical tests on groups.
    
    Parameters:
        data_dict: Dictionary of {group_name: array of values}
        alpha: Significance threshold
    
    Returns:
        Dictionary with test results
    """
    results = {
        'descriptive': {},
        'omnibus': {},
        'posthoc': None
    }
    
    # Descriptive statistics
    for group, values in data_dict.items():
        results['descriptive'][group] = {
            'n': len(values),
            'mean': np.mean(values),
            'std': np.std(values, ddof=1),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Need at least 2 groups with data
    valid_groups = {k: v for k, v in data_dict.items() if len(v) >= 2}
    
    if len(valid_groups) < 2:
        results['omnibus']['error'] = 'Insufficient groups with data for comparison'
        return results
    
    group_values = list(valid_groups.values())
    group_names = list(valid_groups.keys())
    
    # Welch's ANOVA (does not assume equal variances)
    try:
        # Using scipy's one-way ANOVA with equal_var=False equivalent
        # For Welch's ANOVA, we use the Levene test for variance check
        # and then run appropriate ANOVA
        
        # First, check variance homogeneity
        levene_stat, levene_p = stats.levene(*group_values)
        results['omnibus']['levene_statistic'] = levene_stat
        results['omnibus']['levene_p'] = levene_p
        
        # Run standard one-way ANOVA
        f_stat, anova_p = stats.f_oneway(*group_values)
        results['omnibus']['anova_f'] = f_stat
        results['omnibus']['anova_p'] = anova_p
        
        # Kruskal-Wallis (non-parametric alternative)
        h_stat, kw_p = stats.kruskal(*group_values)
        results['omnibus']['kruskal_h'] = h_stat
        results['omnibus']['kruskal_p'] = kw_p
        
    except Exception as e:
        results['omnibus']['error'] = str(e)
        return results
    
    # Post-hoc tests if omnibus is significant
    min_omnibus_p = min(anova_p, kw_p)
    results['omnibus']['significant'] = min_omnibus_p < alpha
    
    if min_omnibus_p < alpha:
        try:
            # Prepare data for post-hoc tests
            all_values = []
            all_groups = []
            for name, values in valid_groups.items():
                all_values.extend(values)
                all_groups.extend([name] * len(values))
            
            posthoc_df = pd.DataFrame({'value': all_values, 'group': all_groups})
            
            # Games-Howell post-hoc (doesn't assume equal variances)
            # Using Dunn's test as it's more commonly available
            posthoc_matrix = sp.posthoc_dunn(posthoc_df, val_col='value', group_col='group', p_adjust='bonferroni')
            
            results['posthoc'] = posthoc_matrix
            
        except Exception as e:
            results['posthoc_error'] = str(e)
    
    return results


def create_comparison_plot(stats_results, title, output_path, 
                           colors={'M': '#2e75b6', 'F': '#c00000', 'All': '#7030a0'}):
    """
    Create a side-by-side bar plot comparing groups across Male, Female, and All.
    
    Parameters:
        stats_results: Dict with keys 'M', 'F', 'All', each containing test results
        title: Plot title
        output_path: Where to save the plot
        colors: Color dictionary for each subset
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    subset_order = ['M', 'F', 'All']
    subset_labels = ['Males', 'Females', 'All Samples']
    
    # Get all bins present across all subsets
    all_bins = set()
    for subset in subset_order:
        if subset in stats_results and 'descriptive' in stats_results[subset]:
            all_bins.update(stats_results[subset]['descriptive'].keys())
    bins = sorted(all_bins)
    
    for idx, (subset, label) in enumerate(zip(subset_order, subset_labels)):
        ax = axes[idx]
        
        if subset not in stats_results or 'descriptive' not in stats_results[subset]:
            ax.set_title(f'{label}\n(No data)')
            ax.set_xticks(range(len(bins)))
            ax.set_xticklabels(bins)
            continue
        
        desc = stats_results[subset]['descriptive']
        omnibus = stats_results[subset].get('omnibus', {})
        posthoc = stats_results[subset].get('posthoc', None)
        
        means = [desc.get(b, {}).get('mean', 0) for b in bins]
        stds = [desc.get(b, {}).get('std', 0) for b in bins]
        ns = [desc.get(b, {}).get('n', 0) for b in bins]
        
        x_pos = np.arange(len(bins))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors[subset], 
                      edgecolor='black', linewidth=1, alpha=0.8)
        
        # Add sample size labels on bars
        for i, (bar, n) in enumerate(zip(bars, ns)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 0.5,
                    f'n={n}', ha='center', va='bottom', fontsize=9)
        
        # Add significance indicators if post-hoc available
        if posthoc is not None and not posthoc.empty:
            # Find significant pairs and add asterisks
            sig_pairs = []
            for i, bin1 in enumerate(bins):
                for j, bin2 in enumerate(bins):
                    if i < j and bin1 in posthoc.index and bin2 in posthoc.columns:
                        p_val = posthoc.loc[bin1, bin2]
                        if p_val < 0.05:
                            sig_pairs.append((i, j, p_val))
            
            # Add significance brackets (simplified - just asterisks above bars)
            y_max = max(m + s for m, s in zip(means, stds)) if means else 0
            for pair_idx, (i, j, p) in enumerate(sig_pairs[:3]):  # Limit to 3 pairs shown
                y_line = y_max + 2 + pair_idx * 2
                asterisk = '***' if p < 0.001 else ('**' if p < 0.01 else '*')
                mid_x = (i + j) / 2
                ax.plot([i, i, j, j], [y_line - 0.5, y_line, y_line, y_line - 0.5], 
                        'k-', linewidth=1)
                ax.text(mid_x, y_line + 0.3, asterisk, ha='center', fontsize=12)
        
        ax.set_xlabel('Bin (Age Group)', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bins)
        ax.set_title(label, fontsize=12, fontweight='bold')
        
        # Add omnibus p-value to subtitle
        if 'anova_p' in omnibus:
            p_str = f"ANOVA p={omnibus['anova_p']:.4f}" if omnibus['anova_p'] >= 0.0001 else "ANOVA p<0.0001"
            ax.text(0.5, -0.15, p_str, transform=ax.transAxes, ha='center', fontsize=9, style='italic')
    
    axes[0].set_ylabel('Call Rate (%)', fontsize=11)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved plot: {output_path}")


def generate_html_report(output_dir, params, sample_stats_all, marker_stats_all,
                         sample_results, marker_results, filtered_samples_df):
    """Generate the HTML report with embedded plots and download links."""
    
    report_path = os.path.join(output_dir, 'report.html')
    
    # Format current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Build HTML content
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Call Rate Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1f618d;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: #c3c3c1;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #c22e2d;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #c22e2d;
            padding-left: 15px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        .summary-box {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .summary-item {{
            margin: 10px 0;
            font-size: 16px;
        }}
        .summary-label {{
            font-weight: bold;
            color: #1f618d;
        }}
        .plot-container {{
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            background-color: #1f618d;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            border-radius: 4px;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .data-table th {{
            background-color: #1f618d;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        .data-table td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
            background-color: white;
        }}
        .data-table tr:hover td {{
            background-color: #f5f5f5;
        }}
        .file-link {{
            display: inline-block;
            margin: 5px 10px 5px 0;
            padding: 8px 15px;
            background-color: #1f618d;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 14px;
        }}
        .file-link:hover {{
            background-color: #2980b9;
        }}
        .methods-box {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            font-size: 14px;
            line-height: 1.6;
        }}
        .significant {{
            color: #c22e2d;
            font-weight: bold;
        }}
        .not-significant {{
            color: #27ae60;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
            color: #7f8c8d;
            font-size: 12px;
            text-align: center;
        }}
        .stat-interpretation {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Call Rate Analysis Report</h1>
        
        <div class="summary-box">
            <div class="summary-item"><span class="summary-label">Report Generated:</span> {timestamp}</div>
            <div class="summary-item"><span class="summary-label">Genotype File:</span> {params['genotype_file']}</div>
            <div class="summary-item"><span class="summary-label">Sample Age Bin Table:</span> {params['breed_bin_file']}</div>
            <div class="summary-item"><span class="summary-label">Output Directory:</span> {params['output_dir']}</div>
            <div class="summary-item"><span class="summary-label">Call Rate Threshold:</span> {params['threshold'] if params['threshold'] else 'None (all samples included)'}</div>
            <div class="summary-item"><span class="summary-label">Exclude Failed Markers:</span> {'Yes' if params['exclude_failed_markers'] else 'No'}</div>
            {f"<div class='summary-item'><span class='summary-label'>Completely Failed Markers Excluded:</span> {params['n_failed_markers']}</div>" if params['exclude_failed_markers'] else ""}
            <div class="summary-item"><span class="summary-label">Total Samples (after filtering):</span> {params['n_samples_after']}</div>
            <div class="summary-item"><span class="summary-label">Samples Excluded:</span> {params['n_samples_excluded']}</div>
            <div class="summary-item"><span class="summary-label">Total Markers:</span> {params['n_markers']}</div>
            <div class="summary-item"><span class="summary-label">Marker Analysis:</span> {'Skipped' if params['skip_marker_analysis'] else 'Performed'}</div>
        </div>
        
        <h2>Methods</h2>
        <div class="methods-box">
            <p><strong>Statistical Approach:</strong></p>
            <p>This analysis compares call rates across four age groups (Bins A, B, C, D) using a two-stage approach:</p>
            <p><strong>Stage 1 - Omnibus Tests:</strong></p>
            <ul>
                <li><strong>Levene's Test:</strong> Assesses equality of variances across groups.</li>
                <li><strong>One-way ANOVA:</strong> Tests whether group means differ significantly (parametric).</li>
                <li><strong>Kruskal-Wallis Test:</strong> Non-parametric alternative that compares group medians.</li>
            </ul>
            <p><strong>Stage 2 - Post-hoc Comparisons:</strong></p>
            <ul>
                <li><strong>Dunn's Test:</strong> Pairwise comparisons following a significant omnibus test, with Bonferroni correction for multiple comparisons.</li>
            </ul>
            <p><strong>Interpretation:</strong> A p-value less than 0.05 indicates statistical significance at the 95% confidence level.</p>
            <p><strong>Significance markers on plots:</strong> * p &lt; 0.05, ** p &lt; 0.01, *** p &lt; 0.001</p>
        </div>
'''
    
    # Sample Call Rate Section
    html += '''
        <h2>Sample Call Rate Analysis</h2>
        <p>Sample call rate represents the percentage of markers with valid genotype calls (AA, BB, or AB) for each sample.</p>
        
        <h3>Results by Group</h3>
        <div class="plot-container">
            <img src="plots/sample_callrate_by_bin.png" alt="Sample Call Rate by Bin">
        </div>
'''
    
    # Add sample call rate statistics tables
    html += _generate_stats_table_html(sample_results, 'Sample Call Rate')
    
    # Marker Call Rate Section - only if not skipped
    if not params['skip_marker_analysis']:
        html += '''
        <h2>Marker Call Rate Analysis</h2>
        <p>Marker call rate represents the percentage of samples with valid genotype calls for each marker.</p>
        
        <h3>Results by Group</h3>
        <div class="plot-container">
            <img src="plots/marker_callrate_by_bin.png" alt="Marker Call Rate by Bin">
        </div>
'''
        
        # Add marker call rate statistics tables
        html += _generate_stats_table_html(marker_results, 'Marker Call Rate')
    
    # Downloads Section
    html += '''
        <h2>Download Data Files</h2>
        
        <h3>Call Rate Tables</h3>
        <div class="summary-box">
            <a href="call_rates/sample_callrates_all.csv" class="file-link">Sample Call Rates (All)</a>
            <a href="call_rates/sample_callrates_males.csv" class="file-link">Sample Call Rates (Males)</a>
            <a href="call_rates/sample_callrates_females.csv" class="file-link">Sample Call Rates (Females)</a>
            <a href="call_rates/sample_callrates_by_bin.csv" class="file-link">Sample Call Rates (By Bin)</a>
'''
    
    if not params['skip_marker_analysis']:
        html += '''
            <br><br>
            <a href="call_rates/marker_callrates_all.csv" class="file-link">Marker Call Rates (All)</a>
            <a href="call_rates/marker_callrates_males.csv" class="file-link">Marker Call Rates (Males)</a>
            <a href="call_rates/marker_callrates_females.csv" class="file-link">Marker Call Rates (Females)</a>
            <a href="call_rates/marker_callrates_by_bin.csv" class="file-link">Marker Call Rates (By Bin)</a>
'''
    
    html += '''
        </div>
        
        <h3>Statistical Results</h3>
        <div class="summary-box">
            <a href="statistics/sample_callrate_stats.csv" class="file-link">Sample Call Rate Statistics</a>
            <a href="statistics/sample_callrate_posthoc.csv" class="file-link">Sample Call Rate Post-hoc</a>
'''
    
    if not params['skip_marker_analysis']:
        html += '''
            <br><br>
            <a href="statistics/marker_callrate_stats.csv" class="file-link">Marker Call Rate Statistics</a>
            <a href="statistics/marker_callrate_posthoc.csv" class="file-link">Marker Call Rate Post-hoc</a>
'''
    
    html += '''
        </div>
        
        <h3>Plots</h3>
        <div class="summary-box">
            <a href="plots/sample_callrate_by_bin.png" class="file-link">Sample Call Rate Plot (PNG)</a>
'''
    
    if not params['skip_marker_analysis']:
        html += '''
            <a href="plots/marker_callrate_by_bin.png" class="file-link">Marker Call Rate Plot (PNG)</a>
'''
    
    html += '''
        </div>
'''
    
    # Filtered samples section (if applicable)
    if filtered_samples_df is not None and len(filtered_samples_df) > 0:
        html += '''
        <h3>Filtered Samples</h3>
        <div class="summary-box">
            <a href="filtered_samples.csv" class="file-link">Excluded Samples List</a>
        </div>
'''
    
    # Footer
    html += '''
        <div class="footer">
            <p>Generated by Call Rate Analysis Program</p>
            <p>END OF REPORT</p>
        </div>
    </div>
</body>
</html>
'''
    
    with open(report_path, 'w') as f:
        f.write(html)
    
    print(f"  Generated HTML report: {report_path}")


def _generate_stats_table_html(results, analysis_type):
    """Generate HTML tables for statistical results."""
    html = ''
    
    subset_labels = {'M': 'Males', 'F': 'Females', 'All': 'All Samples'}
    
    for subset in ['M', 'F', 'All']:
        if subset not in results:
            continue
            
        res = results[subset]
        label = subset_labels[subset]
        
        html += f'''
        <h3>{label}</h3>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Bin</th>
                    <th>N</th>
                    <th>Mean (%)</th>
                    <th>Std Dev</th>
                    <th>Median (%)</th>
                    <th>Min (%)</th>
                    <th>Max (%)</th>
                </tr>
            </thead>
            <tbody>
'''
        
        if 'descriptive' in res:
            for bin_name in sorted(res['descriptive'].keys()):
                stats = res['descriptive'][bin_name]
                html += f'''
                <tr>
                    <td><strong>{bin_name}</strong></td>
                    <td>{stats['n']}</td>
                    <td>{stats['mean']:.2f}</td>
                    <td>{stats['std']:.2f}</td>
                    <td>{stats['median']:.2f}</td>
                    <td>{stats['min']:.2f}</td>
                    <td>{stats['max']:.2f}</td>
                </tr>
'''
        
        html += '''
            </tbody>
        </table>
'''
        
        # Omnibus test results
        if 'omnibus' in res and 'anova_p' in res['omnibus']:
            omnibus = res['omnibus']
            anova_sig = 'significant' if omnibus['anova_p'] < 0.05 else 'not-significant'
            kw_sig = 'significant' if omnibus['kruskal_p'] < 0.05 else 'not-significant'
            
            html += f'''
        <div class="stat-interpretation">
            <strong>Omnibus Test Results ({label}):</strong><br>
            Levene's Test (variance equality): F = {omnibus['levene_statistic']:.4f}, p = {omnibus['levene_p']:.4f}<br>
            One-way ANOVA: F = {omnibus['anova_f']:.4f}, p = <span class="{anova_sig}">{omnibus['anova_p']:.4f}</span><br>
            Kruskal-Wallis: H = {omnibus['kruskal_h']:.4f}, p = <span class="{kw_sig}">{omnibus['kruskal_p']:.4f}</span>
        </div>
'''
        
        # Post-hoc results
        if res.get('posthoc') is not None and not res['posthoc'].empty:
            html += f'''
        <h4>Post-hoc Pairwise Comparisons (Dunn's Test with Bonferroni correction)</h4>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Comparison</th>
                    <th>p-value</th>
                    <th>Significant</th>
                </tr>
            </thead>
            <tbody>
'''
            posthoc = res['posthoc']
            bins = sorted(posthoc.index)
            for i, bin1 in enumerate(bins):
                for bin2 in bins[i+1:]:
                    p_val = posthoc.loc[bin1, bin2]
                    sig_class = 'significant' if p_val < 0.05 else 'not-significant'
                    sig_text = 'Yes' if p_val < 0.05 else 'No'
                    html += f'''
                <tr>
                    <td>{bin1} vs {bin2}</td>
                    <td class="{sig_class}">{p_val:.4f}</td>
                    <td class="{sig_class}">{sig_text}</td>
                </tr>
'''
            html += '''
            </tbody>
        </table>
'''
    
    return html


def save_statistics_csv(results, output_dir, prefix):
    """Save statistical results to CSV files."""
    stats_dir = os.path.join(output_dir, 'statistics')
    
    # Descriptive statistics
    rows = []
    for subset in ['M', 'F', 'All']:
        if subset not in results:
            continue
        subset_label = {'M': 'Males', 'F': 'Females', 'All': 'All'}[subset]
        
        if 'descriptive' in results[subset]:
            for bin_name, stats in results[subset]['descriptive'].items():
                row = {
                    'Subset': subset_label,
                    'Bin': bin_name,
                    'N': stats['n'],
                    'Mean': stats['mean'],
                    'Std_Dev': stats['std'],
                    'Median': stats['median'],
                    'Min': stats['min'],
                    'Max': stats['max']
                }
                
                # Add omnibus test results
                if 'omnibus' in results[subset]:
                    omnibus = results[subset]['omnibus']
                    row['ANOVA_F'] = omnibus.get('anova_f', '')
                    row['ANOVA_p'] = omnibus.get('anova_p', '')
                    row['Kruskal_H'] = omnibus.get('kruskal_h', '')
                    row['Kruskal_p'] = omnibus.get('kruskal_p', '')
                
                rows.append(row)
    
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(stats_dir, f'{prefix}_stats.csv'), index=False)
    
    # Post-hoc results
    posthoc_rows = []
    for subset in ['M', 'F', 'All']:
        if subset not in results:
            continue
        subset_label = {'M': 'Males', 'F': 'Females', 'All': 'All'}[subset]
        
        if results[subset].get('posthoc') is not None and not results[subset]['posthoc'].empty:
            posthoc = results[subset]['posthoc']
            bins = sorted(posthoc.index)
            for i, bin1 in enumerate(bins):
                for bin2 in bins[i+1:]:
                    posthoc_rows.append({
                        'Subset': subset_label,
                        'Group1': bin1,
                        'Group2': bin2,
                        'p_value': posthoc.loc[bin1, bin2],
                        'Significant': 'Yes' if posthoc.loc[bin1, bin2] < 0.05 else 'No'
                    })
    
    posthoc_df = pd.DataFrame(posthoc_rows)
    if posthoc_df.empty:
        posthoc_df = pd.DataFrame(columns=['Subset', 'Group1', 'Group2', 'p_value', 'Significant'])
    posthoc_df.to_csv(os.path.join(stats_dir, f'{prefix}_posthoc.csv'), index=False)


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Validate inputs
    if not os.path.exists(args.genotype_file):
        print(f"Error: Genotype file not found: {args.genotype_file}")
        sys.exit(1)
    
    if not os.path.exists(args.breed_bin_table):
        print(f"Error: Breed/Bin table not found: {args.breed_bin_table}")
        sys.exit(1)
    
    if args.callrate is not None and (args.callrate < 0 or args.callrate > 100):
        print(f"Error: Call rate threshold must be between 0 and 100")
        sys.exit(1)
    
    # Create output directory structure
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'call_rates'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'statistics'), exist_ok=True)
    
    print("\n" + "="*60)
    print("CALL RATE ANALYSIS")
    print("="*60)
    
    # Load data
    genotype_df = load_genotype_data(args.genotype_file)
    breed_bin_df = load_breed_bin_table(args.breed_bin_table)
    
    # Exclude completely failed markers if requested
    failed_markers_list = []
    if args.exclude_failed_markers:
        genotype_df, failed_markers_list = exclude_failed_markers(genotype_df)
    
    # Calculate initial sample call rates for filtering
    print("\nCalculating initial sample call rates...")
    initial_sample_rates = calculate_sample_call_rates(genotype_df)
    
    # Apply threshold filter if specified
    filtered_samples_df = None
    n_samples_excluded = 0
    
    if args.callrate is not None:
        print(f"\nApplying call rate threshold: {args.callrate}%")
        passing_samples, filtered_samples_df = filter_samples_by_callrate(
            initial_sample_rates, args.callrate
        )
        n_samples_excluded = len(filtered_samples_df)
        print(f"  Samples passing threshold: {len(passing_samples)}")
        print(f"  Samples excluded: {n_samples_excluded}")
        
        # Filter genotype data
        genotype_df = genotype_df[genotype_df['Sample'].isin(passing_samples)]
        
        # Save filtered samples list
        if n_samples_excluded > 0:
            filtered_samples_df.to_csv(os.path.join(output_dir, 'filtered_samples.csv'), index=False)
    
    # Merge breed/bin info with samples
    samples_in_data = genotype_df['Sample'].unique()
    breed_bin_df = breed_bin_df[breed_bin_df['Sample'].isin(samples_in_data)]
    
    # Identify samples by group
    males = breed_bin_df[breed_bin_df['Breed'] == 'M']['Sample'].tolist()
    females = breed_bin_df[breed_bin_df['Breed'] == 'F']['Sample'].tolist()
    all_samples = breed_bin_df['Sample'].tolist()
    
    print(f"\nSample breakdown after filtering:")
    print(f"  Males: {len(males)}")
    print(f"  Females: {len(females)}")
    print(f"  Unknown: {len(all_samples) - len(males) - len(females)}")
    print(f"  Total: {len(all_samples)}")
    
    # ===== SAMPLE CALL RATE ANALYSIS =====
    print("\n" + "-"*40)
    print("SAMPLE CALL RATE ANALYSIS")
    print("-"*40)
    
    # Calculate sample call rates
    sample_call_rates = calculate_sample_call_rates(genotype_df)
    sample_call_rates = sample_call_rates.merge(breed_bin_df, on='Sample', how='left')
    
    # Save call rate tables
    sample_call_rates.to_csv(os.path.join(output_dir, 'call_rates', 'sample_callrates_all.csv'), index=False)
    sample_call_rates[sample_call_rates['Breed'] == 'M'].to_csv(
        os.path.join(output_dir, 'call_rates', 'sample_callrates_males.csv'), index=False)
    sample_call_rates[sample_call_rates['Breed'] == 'F'].to_csv(
        os.path.join(output_dir, 'call_rates', 'sample_callrates_females.csv'), index=False)
    sample_call_rates.to_csv(os.path.join(output_dir, 'call_rates', 'sample_callrates_by_bin.csv'), index=False)
    
    # Run statistical tests for sample call rates
    sample_results = {}
    
    for subset, subset_name in [('M', 'Males'), ('F', 'Females'), ('All', 'All')]:
        print(f"\n  Analyzing {subset_name}...")
        
        if subset == 'All':
            subset_df = sample_call_rates
        else:
            subset_df = sample_call_rates[sample_call_rates['Breed'] == subset]
        
        # Group by Bin
        data_dict = {}
        for bin_name in subset_df['Bin'].dropna().unique():
            values = subset_df[subset_df['Bin'] == bin_name]['Call_Rate'].values
            if len(values) > 0:
                data_dict[bin_name] = values
        
        if data_dict:
            sample_results[subset] = run_statistical_tests(data_dict)
            print(f"    Bins found: {list(data_dict.keys())}")
    
    # Create sample call rate plot
    print("\n  Generating sample call rate plot...")
    create_comparison_plot(
        sample_results, 
        'Sample Call Rate by Age Group',
        os.path.join(output_dir, 'plots', 'sample_callrate_by_bin.png')
    )
    
    # Save statistics
    save_statistics_csv(sample_results, output_dir, 'sample_callrate')
    
    # ===== MARKER CALL RATE ANALYSIS =====
    marker_results = {}
    marker_callrates_by_bin = pd.DataFrame()
    
    if not args.skip_marker_analysis:
        print("\n" + "-"*40)
        print("MARKER CALL RATE ANALYSIS")
        print("-"*40)
        
        # Calculate marker call rates for each subset
        marker_call_rates_all = calculate_marker_call_rates(genotype_df)
        marker_call_rates_all.to_csv(os.path.join(output_dir, 'call_rates', 'marker_callrates_all.csv'), index=False)
        
        # Males
        marker_cr_males = calculate_marker_call_rates(genotype_df, males)
        marker_cr_males.to_csv(os.path.join(output_dir, 'call_rates', 'marker_callrates_males.csv'), index=False)
        
        # Females
        marker_cr_females = calculate_marker_call_rates(genotype_df, females)
        marker_cr_females.to_csv(os.path.join(output_dir, 'call_rates', 'marker_callrates_females.csv'), index=False)
        
        # By Bin - need to calculate marker call rates for each bin
        bins = breed_bin_df['Bin'].dropna().unique()
        marker_by_bin_rows = []
        
        for bin_name in bins:
            bin_samples = breed_bin_df[breed_bin_df['Bin'] == bin_name]['Sample'].tolist()
            bin_marker_rates = calculate_marker_call_rates(genotype_df, bin_samples)
            bin_marker_rates['Bin'] = bin_name
            marker_by_bin_rows.append(bin_marker_rates)
        
        if marker_by_bin_rows:
            marker_callrates_by_bin = pd.concat(marker_by_bin_rows, ignore_index=True)
            marker_callrates_by_bin.to_csv(os.path.join(output_dir, 'call_rates', 'marker_callrates_by_bin.csv'), index=False)
        
        # Statistical tests for marker call rates
        for subset, subset_name in [('M', 'Males'), ('F', 'Females'), ('All', 'All')]:
            print(f"\n  Analyzing {subset_name}...")
            
            data_dict = {}
            for bin_name in bins:
                if subset == 'All':
                    bin_samples = breed_bin_df[breed_bin_df['Bin'] == bin_name]['Sample'].tolist()
                else:
                    bin_samples = breed_bin_df[(breed_bin_df['Bin'] == bin_name) & 
                                               (breed_bin_df['Breed'] == subset)]['Sample'].tolist()
                
                if bin_samples:
                    marker_rates = calculate_marker_call_rates(genotype_df, bin_samples)
                    data_dict[bin_name] = marker_rates['Call_Rate'].values
            
            if data_dict:
                marker_results[subset] = run_statistical_tests(data_dict)
                print(f"    Bins found: {list(data_dict.keys())}")
        
        # Create marker call rate plot
        print("\n  Generating marker call rate plot...")
        create_comparison_plot(
            marker_results,
            'Marker Call Rate by Age Group',
            os.path.join(output_dir, 'plots', 'marker_callrate_by_bin.png')
        )
        
        # Save statistics
        save_statistics_csv(marker_results, output_dir, 'marker_callrate')
    else:
        print("\n" + "-"*40)
        print("MARKER CALL RATE ANALYSIS - SKIPPED")
        print("-"*40)
    
    # ===== GENERATE HTML REPORT =====
    print("\n" + "-"*40)
    print("GENERATING HTML REPORT")
    print("-"*40)
    
    params = {
        'genotype_file': os.path.basename(args.genotype_file),
        'breed_bin_file': os.path.basename(args.breed_bin_table),
        'output_dir': output_dir,
        'threshold': args.callrate,
        'n_samples_after': len(all_samples),
        'n_samples_excluded': n_samples_excluded,
        'n_markers': genotype_df['Marker'].nunique(),
        'exclude_failed_markers': args.exclude_failed_markers,
        'n_failed_markers': len(failed_markers_list),
        'skip_marker_analysis': args.skip_marker_analysis
    }
    
    generate_html_report(
        output_dir, params, sample_call_rates, marker_callrates_by_bin,
        sample_results, marker_results, filtered_samples_df
    )
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Open {os.path.join(output_dir, 'report.html')} to view results")


if __name__ == '__main__':
    main()
