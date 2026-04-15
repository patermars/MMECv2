"""Generate dataset visualization graphs for research paper."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import torch
from datetime import datetime

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

def load_data():
    """Load processed data and labels."""
    labels_path = Path('data/processed/labels.csv')
    processed_dir = Path('data/processed')
    
    labels = pd.read_csv(labels_path)
    labels['date'] = pd.to_datetime(labels['call_date'])
    
    # Load feature dimensions from processed files
    features = []
    print(f"Scanning {processed_dir} for .pt files...")
    for pt_file in list(processed_dir.glob('*.pt'))[:10]:  # Sample first 10 for speed
        try:
            data = torch.load(pt_file, map_location='cpu')
            features.append({
                'call_id': pt_file.stem,
                'n_utterances': len(data['acoustic_features']),
                'acoustic_dim': data['acoustic_features'].shape[1] if len(data['acoustic_features']) > 0 else 0,
                'text_dim': data['text_embeddings'].shape[1] if len(data['text_embeddings']) > 0 else 0,
                'audio_dim': data['audio_embeddings'].shape[1] if len(data['audio_embeddings']) > 0 else 0,
            })
        except Exception as e:
            continue
    
    if features:
        features_df = pd.DataFrame(features)
        print(f"Loaded {len(features_df)} feature files")
        merged = labels.merge(features_df, on='call_id', how='inner')
    else:
        # If no .pt files, use labels only with default dimensions
        print("No .pt files found, using labels only with default dimensions")
        merged = labels.copy()
        merged['n_utterances'] = 150  # Default estimate
        merged['acoustic_dim'] = 29
        merged['text_dim'] = 768
        merged['audio_dim'] = 768
    
    return merged

def plot_temporal_distribution(df, save_path='figures/temporal_distribution.png'):
    """Plot temporal distribution of earnings calls."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), height_ratios=[2, 1])
    
    # Timeline with split markers
    df_sorted = df.sort_values('date')
    ax1.scatter(df_sorted['date'], df_sorted.index, alpha=0.6, s=20, c='#6366f1')
    
    # Add split lines
    train_end = pd.to_datetime('2017-10-31')
    val_end = pd.to_datetime('2017-12-31')
    ax1.axvline(train_end, color='#ef4444', linestyle='--', linewidth=1.5, label='Train/Val Split')
    ax1.axvline(val_end, color='#f59e0b', linestyle='--', linewidth=1.5, label='Val/Test Split')
    
    ax1.set_ylabel('Call Index')
    ax1.set_title('Temporal Distribution of Earnings Calls (N={})'.format(len(df)))
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(alpha=0.3)
    
    # Monthly histogram
    df['month'] = df['date'].dt.to_period('M')
    monthly_counts = df.groupby('month').size()
    ax2.bar(range(len(monthly_counts)), monthly_counts.values, color='#6366f1', alpha=0.7)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Count')
    ax2.set_xticks(range(len(monthly_counts)))
    ax2.set_xticklabels([str(m) for m in monthly_counts.index], rotation=45, ha='right')
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_volatility_distributions(df, save_path='figures/volatility_distributions.png'):
    """Plot distributions of target volatility across time windows."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    
    windows = ['abnormal_vol_1d', 'abnormal_vol_3d', 'abnormal_vol_7d']
    titles = ['1-Day', '3-Day', '7-Day']
    
    for ax, col, title in zip(axes, windows, titles):
        data = df[col].dropna()
        ax.hist(data, bins=40, color='#6366f1', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axvline(data.median(), color='#ef4444', linestyle='--', linewidth=1.5, label=f'Median: {data.median():.4f}')
        ax.set_xlabel('Abnormal Volatility')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{title} Post-Call Volatility')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_feature_dimensions(df, save_path='figures/feature_dimensions.png'):
    """Plot distribution of utterance counts and feature dimensions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    
    # Utterance distribution
    ax = axes[0]
    ax.hist(df['n_utterances'], bins=30, color='#10b981', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(df['n_utterances'].median(), color='#ef4444', linestyle='--', linewidth=1.5, 
               label=f'Median: {df["n_utterances"].median():.0f}')
    ax.axvline(200, color='#f59e0b', linestyle='--', linewidth=1.5, label='Max (truncated): 200')
    ax.set_xlabel('Number of Utterances per Call')
    ax.set_ylabel('Frequency')
    ax.set_title('Utterance Count Distribution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    
    # Feature dimensions bar chart
    ax = axes[1]
    dims = {
        'Acoustic\n(Praat+librosa)': df['acoustic_dim'].iloc[0],
        'Text\n(FinBERT)': df['text_dim'].iloc[0],
        'Audio\n(wav2vec2)': df['audio_dim'].iloc[0]
    }
    bars = ax.bar(dims.keys(), dims.values(), color=['#6366f1', '#10b981', '#f59e0b'], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Embedding Dimension')
    ax.set_title('Feature Dimensions per Utterance')
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}D', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_volatility_correlation(df, save_path='figures/volatility_correlation.png'):
    """Plot correlation between different volatility windows."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    
    pairs = [
        ('abnormal_vol_1d', 'abnormal_vol_3d', '1-Day vs 3-Day'),
        ('abnormal_vol_1d', 'abnormal_vol_7d', '1-Day vs 7-Day'),
        ('abnormal_vol_3d', 'abnormal_vol_7d', '3-Day vs 7-Day')
    ]
    
    for ax, (x_col, y_col, title) in zip(axes, pairs):
        data = df[[x_col, y_col]].dropna()
        ax.scatter(data[x_col], data[y_col], alpha=0.4, s=15, c='#6366f1')
        
        # Add correlation coefficient
        corr = data[x_col].corr(data[y_col])
        ax.text(0.05, 0.95, f'ρ = {corr:.3f}', transform=ax.transAxes, 
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(x_col.replace('abnormal_vol_', '').replace('d', '-Day'))
        ax.set_ylabel(y_col.replace('abnormal_vol_', '').replace('d', '-Day'))
        ax.set_title(title)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_market_model_quality(df, save_path='figures/market_model_quality.png'):
    """Plot market model estimation quality metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    
    # R-squared distribution
    ax = axes[0]
    ax.hist(df['r2'], bins=30, color='#6366f1', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(df['r2'].median(), color='#ef4444', linestyle='--', linewidth=1.5,
               label=f'Median: {df["r2"].median():.3f}')
    ax.set_xlabel('R² (Market Model Fit)')
    ax.set_ylabel('Frequency')
    ax.set_title('Market Model Quality')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    
    # Beta distribution
    ax = axes[1]
    ax.hist(df['beta'], bins=30, color='#10b981', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(1.0, color='#ef4444', linestyle='--', linewidth=1.5, label='Market β = 1.0')
    ax.axvline(df['beta'].median(), color='#f59e0b', linestyle='--', linewidth=1.5,
               label=f'Median: {df["beta"].median():.3f}')
    ax.set_xlabel('Beta (Market Sensitivity)')
    ax.set_ylabel('Frequency')
    ax.set_title('Beta Distribution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    
    # Alpha distribution
    ax = axes[2]
    ax.hist(df['alpha'] * 100, bins=30, color='#f59e0b', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='#ef4444', linestyle='--', linewidth=1.5, label='α = 0')
    ax.set_xlabel('Alpha (%) - Daily Excess Return')
    ax.set_ylabel('Frequency')
    ax.set_title('Alpha Distribution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_sector_analysis(df, save_path='figures/sector_analysis.png'):
    """Plot top sectors and their volatility characteristics."""
    # Extract sector from ticker (simplified - you may need to load actual sector data)
    # For now, show top tickers
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Top tickers by call count
    ax = axes[0]
    top_tickers = df['ticker'].value_counts().head(15)
    ax.barh(range(len(top_tickers)), top_tickers.values, color='#6366f1', alpha=0.7)
    ax.set_yticks(range(len(top_tickers)))
    ax.set_yticklabels(top_tickers.index)
    ax.set_xlabel('Number of Calls')
    ax.set_title('Top 15 Companies by Call Count')
    ax.grid(alpha=0.3, axis='x')
    
    # Volatility by top tickers
    ax = axes[1]
    top_ticker_list = top_tickers.head(10).index
    vol_by_ticker = df[df['ticker'].isin(top_ticker_list)].groupby('ticker')['abnormal_vol_3d'].mean().sort_values()
    colors = ['#10b981' if v < vol_by_ticker.median() else '#ef4444' for v in vol_by_ticker.values]
    ax.barh(range(len(vol_by_ticker)), vol_by_ticker.values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(vol_by_ticker)))
    ax.set_yticklabels(vol_by_ticker.index)
    ax.set_xlabel('Mean 3-Day Abnormal Volatility')
    ax.set_title('Average Volatility by Company')
    ax.axvline(vol_by_ticker.median(), color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

def plot_summary_statistics(df, save_path='figures/summary_table.png'):
    """Create summary statistics table as figure."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Compute statistics
    stats = {
        'Metric': [
            'Total Calls',
            'Date Range',
            'Unique Companies',
            'Avg Utterances/Call',
            'Median 1-Day Vol',
            'Median 3-Day Vol',
            'Median 7-Day Vol',
            'Mean Market Beta',
            'Mean R²'
        ],
        'Value': [
            f"{len(df):,}",
            f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
            f"{df['ticker'].nunique():,}",
            f"{df['n_utterances'].mean():.1f} ± {df['n_utterances'].std():.1f}",
            f"{df['abnormal_vol_1d'].median():.4f}",
            f"{df['abnormal_vol_3d'].median():.4f}",
            f"{df['abnormal_vol_7d'].median():.4f}",
            f"{df['beta'].mean():.3f} ± {df['beta'].std():.3f}",
            f"{df['r2'].mean():.3f} ± {df['r2'].std():.3f}"
        ]
    }
    
    table = ax.table(cellText=[[stats['Metric'][i], stats['Value'][i]] for i in range(len(stats['Metric']))],
                     colLabels=['Metric', 'Value'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.6, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#6366f1')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(stats['Metric']) + 1):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Dataset Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    plt.close()

def main():
    """Generate all visualization graphs."""
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} calls with complete data\n")
    
    print("Generating visualizations...")
    plot_temporal_distribution(df)
    plot_volatility_distributions(df)
    plot_feature_dimensions(df)
    plot_volatility_correlation(df)
    plot_market_model_quality(df)
    plot_sector_analysis(df)
    plot_summary_statistics(df)
    
    print("\n✓ All visualizations saved to figures/")
    print("\nFigures generated:")
    print("  1. temporal_distribution.png - Timeline and monthly distribution")
    print("  2. volatility_distributions.png - Target variable distributions")
    print("  3. feature_dimensions.png - Utterance counts and embedding dimensions")
    print("  4. volatility_correlation.png - Cross-window correlations")
    print("  5. market_model_quality.png - R², beta, alpha distributions")
    print("  6. sector_analysis.png - Company-level statistics")
    print("  7. summary_table.png - Dataset summary table")

if __name__ == '__main__':
    main()
