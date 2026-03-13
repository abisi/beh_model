#!/usr/bin/env python3
"""
Simple function to check multicollinearity using session-averaged features.

@author: Axel Bisi
@project: beh_model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


def check_multicollinearity(data, features):
    """
    Check multicollinearity at both trial and session levels.
    
    Averages features within (mouse_id, session_id) to remove temporal
    autocorrelation and reveal true feature redundancy.
    
    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns: 'mouse_id', 'session_id', and all features
    features : list of str
        Feature names to analyze (exclude 'bias' if it's constant)
    
    Returns
    -------
    dict with keys:
        'trial_corr' : pd.DataFrame - Trial-level correlation matrix
        'session_corr' : pd.DataFrame - Session-level correlation matrix
        'trial_vif' : pd.DataFrame - Trial-level VIF values
        'session_vif' : pd.DataFrame - Session-level VIF values
        'high_corr_pairs' : pd.DataFrame - Pairs with |r| > 0.7 at session level
        'summary' : dict - Summary statistics
    
    Examples
    --------
    >>> results = check_multicollinearity(data_train, 
    ...     ['whisker', 'auditory', 'time_since_last_whisker_stim'])
    >>> print(results['summary']['interpretation'])
    >>> print(results['high_corr_pairs'])
    """
    
    # Validate inputs
    required_cols = ['mouse_id', 'session_id'] + features
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # 1. Session-averaged features
    session_averaged = data.groupby(['mouse_id', 'session_id'])[features].mean().reset_index()
    
    # 2. Trial-level correlations and VIF
    trial_corr = data[features].corr()
    trial_vif = _compute_vif(data, features)
    
    # 3. Session-level correlations and VIF
    session_corr = session_averaged[features].corr()
    session_vif = _compute_vif(session_averaged, features)
    
    # 4. Find high correlation pairs at session level
    high_corr_pairs = _find_high_correlations(session_corr, threshold=0.7)
    
    # 5. Summary statistics
    max_trial_vif = trial_vif['VIF'].max()
    max_session_vif = session_vif['VIF'].max()
    max_session_corr = high_corr_pairs['abs_correlation'].max() if len(high_corr_pairs) > 0 else 0
    
    # Interpretation
    if max_session_vif > 10 or max_session_corr > 0.7:
        interpretation = "SEVERE multicollinearity at session level - features are truly redundant"
        recommendation = "Remove one feature from each high-correlation pair, or use very strong regularization (prior_sigma=0.1-0.3)"
    elif max_session_vif > 5 or max_session_corr > 0.5:
        interpretation = "MODERATE multicollinearity at session level"
        recommendation = "Use moderate regularization (prior_sigma=0.5) or consider feature selection"
    else:
        interpretation = "LOW multicollinearity at session level - features are not fundamentally redundant"
        if max_trial_vif > 10:
            recommendation = "High trial-level correlations are due to temporal autocorrelation (OK). Use moderate regularization (prior_sigma=0.5-1.0)"
        else:
            recommendation = "No multicollinearity issues. Standard regularization is fine (prior_sigma=1.0-2.0)"
    
    summary = {
        'n_trials': len(data),
        'n_sessions': len(session_averaged),
        'n_mice': data['mouse_id'].nunique(),
        'n_features': len(features),
        'max_trial_vif': max_trial_vif,
        'max_session_vif': max_session_vif,
        'max_session_corr': max_session_corr,
        'n_high_corr_pairs': len(high_corr_pairs),
        'interpretation': interpretation,
        'recommendation': recommendation
    }
    
    return {
        'trial_corr': trial_corr,
        'session_corr': session_corr,
        'trial_vif': trial_vif,
        'session_vif': session_vif,
        'high_corr_pairs': high_corr_pairs,
        'summary': summary
    }


def _compute_vif(data, features):
    """Compute Variance Inflation Factor for each feature."""
    vif_results = []
    
    for target_feature in features:
        predictor_features = [f for f in features if f != target_feature]
        
        if len(predictor_features) == 0:
            vif_results.append({
                'feature': target_feature,
                'VIF': 1.0,
                'R_squared': 0.0
            })
            continue
        
        X = data[predictor_features].values
        y = data[target_feature].values
        
        # Check for missing/infinite values
        if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)):
            vif_results.append({
                'feature': target_feature,
                'VIF': np.nan,
                'R_squared': np.nan
            })
            continue
        
        try:
            model = LinearRegression()
            model.fit(X, y)
            r_squared = model.score(X, y)
            
            vif = 1.0 / (1.0 - r_squared) if r_squared < 0.999 else np.inf
            
            vif_results.append({
                'feature': target_feature,
                'VIF': vif,
                'R_squared': r_squared
            })
        except:
            vif_results.append({
                'feature': target_feature,
                'VIF': np.nan,
                'R_squared': np.nan
            })
    
    return pd.DataFrame(vif_results)


def _find_high_correlations(corr_matrix, threshold=0.7):
    """Find feature pairs with |correlation| > threshold."""
    high_corr_pairs = []
    features = corr_matrix.columns.tolist()
    
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > threshold:
                high_corr_pairs.append({
                    'feature1': features[i],
                    'feature2': features[j],
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })
    
    df = pd.DataFrame(high_corr_pairs)
    if len(df) > 0:
        df = df.sort_values('abs_correlation', ascending=False)
    
    return df


def print_results(results):
    """
    Pretty-print the multicollinearity analysis results.
    
    Parameters
    ----------
    results : dict
        Output from check_multicollinearity()
    """
    print('='*70)
    print('MULTICOLLINEARITY ANALYSIS')
    print('='*70)
    
    # Summary
    s = results['summary']
    print(f"\nDataset: {s['n_trials']} trials, {s['n_sessions']} sessions, {s['n_mice']} mice")
    print(f"Features analyzed: {s['n_features']}")
    
    print('\n' + '-'*70)
    print('TRIAL-LEVEL (includes temporal autocorrelation)')
    print('-'*70)
    print(f"Max VIF: {s['max_trial_vif']:.2f}")
    print("\nCorrelation matrix:")
    print(results['trial_corr'].round(3))
    
    print('\n' + '-'*70)
    print('SESSION-LEVEL (true multicollinearity)')
    print('-'*70)
    print(f"Max VIF: {s['max_session_vif']:.2f}")
    print(f"Max |correlation|: {s['max_session_corr']:.3f}")
    print("\nCorrelation matrix:")
    print(results['session_corr'].round(3))
    
    if len(results['high_corr_pairs']) > 0:
        print(f"\n⚠️  {s['n_high_corr_pairs']} high correlation pairs (|r| > 0.7):")
        print(results['high_corr_pairs'][['feature1', 'feature2', 'correlation']].to_string(index=False))
    else:
        print("\n✓ No high correlations at session level")
    
    print('\n' + '-'*70)
    print('VIF COMPARISON')
    print('-'*70)
    print("\nTrial-level VIF:")
    print(results['trial_vif'].sort_values('VIF', ascending=False).to_string(index=False))
    print("\nSession-level VIF:")
    print(results['session_vif'].sort_values('VIF', ascending=False).to_string(index=False))
    
    print('\n' + '='*70)
    print('INTERPRETATION')
    print('='*70)
    print(f"\n{s['interpretation']}")
    print(f"\nRECOMMENDATION:")
    print(f"{s['recommendation']}")
    print('='*70)


def plot_multicollinearity(results, figsize=(30, 20), save_path=None):
    """
    Create comprehensive visualization of multicollinearity analysis.
    
    Parameters
    ----------
    results : dict
        Output from check_multicollinearity()
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str or Path, optional
        If provided, save figure to this path (without extension)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    
    Examples
    --------
    >>> results = check_multicollinearity(data, features)
    >>> fig = plot_multicollinearity(results)
    >>> plt.show()
    >>> 
    >>> # Or save it
    >>> plot_multicollinearity(results, save_path='multicollinearity_analysis')
    """
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # Color scheme
    cmap = 'coolwarm'
    
    # Extract data
    trial_corr = results['trial_corr']
    session_corr = results['session_corr']
    trial_vif = results['trial_vif']
    session_vif = results['session_vif']
    high_corr = results['high_corr_pairs']
    summary = results['summary']
    
    # 1. Trial-level correlation matrix
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(trial_corr, annot=True, fmt='.2f', cmap=cmap, center=0,
                square=True, ax=ax1, cbar_kws={'label': 'Correlation'},
                vmin=-1, vmax=1, linewidths=0.5, linecolor='gray',
                annot_kws={'fontsize': 8})
    ax1.set_title('Trial-Level Correlations\n(includes temporal autocorr.)',
                 fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.tick_params(axis='both', labelsize=8)
    
    # 2. Session-level correlation matrix
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(session_corr, annot=True, fmt='.2f', cmap=cmap, center=0,
                square=True, ax=ax2, cbar_kws={'label': 'Correlation'},
                vmin=-1, vmax=1, linewidths=0.5, linecolor='gray',
                annot_kws={'fontsize': 8})
    ax2.set_title('Session-Level Correlations\n(true redundancy)',
                 fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    ax2.tick_params(axis='both', labelsize=8)
    
    # Add warning if high correlations
    if len(high_corr) > 0:
        ax2.text(0.02, 0.98, f'⚠️ {len(high_corr)} high\ncorrelation pairs',
                transform=ax2.transAxes, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                fontsize=9)
    else:
        ax2.text(0.02, 0.98, '✓ No high\ncorrelations',
                transform=ax2.transAxes, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                fontsize=9)
    
    # 3. Correlation change (Trial - Session)
    ax3 = fig.add_subplot(gs[0, 2])
    corr_diff = trial_corr - session_corr
    sns.heatmap(corr_diff, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0,
                square=True, ax=ax3, cbar_kws={'label': 'Reduction'},
                vmin=-0.5, vmax=1, linewidths=0.5, linecolor='gray',
                annot_kws={'fontsize': 8})
    ax3.set_title('Correlation Reduction\n(temporal autocorr. removed)',
                 fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel('')
    ax3.set_ylabel('')
    ax3.tick_params(axis='both', labelsize=8)
    
    # 4. VIF comparison
    ax4 = fig.add_subplot(gs[1, :2])
    
    # Prepare data for grouped bar plot
    trial_vif_sorted = trial_vif.sort_values('VIF', ascending=False)
    session_vif_sorted = session_vif.set_index('feature').loc[trial_vif_sorted['feature']].reset_index()
    
    x_pos = np.arange(len(trial_vif_sorted))
    width = 0.35
    
    # Color bars by severity
    trial_colors = ['red' if vif > 10 else 'orange' if vif > 5 else 'green'
                   for vif in trial_vif_sorted['VIF']]
    session_colors = ['red' if vif > 10 else 'orange' if vif > 5 else 'green'
                     for vif in session_vif_sorted['VIF']]
    
    bars1 = ax4.bar(x_pos - width/2, trial_vif_sorted['VIF'], width,
                   label='Trial-level', alpha=0.8, color=trial_colors, edgecolor='black')
    bars2 = ax4.bar(x_pos + width/2, session_vif_sorted['VIF'], width,
                   label='Session-level', alpha=0.8, color=session_colors, edgecolor='black')
    
    ax4.axhline(y=5, color='orange', linestyle='--', linewidth=1.5, 
               alpha=0.7, label='VIF=5 (moderate)')
    ax4.axhline(y=10, color='red', linestyle='--', linewidth=1.5,
               alpha=0.7, label='VIF=10 (severe)')
    
    ax4.set_ylabel('VIF', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax4.set_title('Variance Inflation Factor Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(trial_vif_sorted['feature'], rotation=45, ha='right', fontsize=9)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(bottom=0)
    
    # 5. Summary statistics panel
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    # Determine status
    if summary['max_session_vif'] > 10 or summary['max_session_corr'] > 0.7:
        status_color = 'red'
        status_text = '⚠️ SEVERE\nMULTICOLLINEARITY'
    elif summary['max_session_vif'] > 5 or summary['max_session_corr'] > 0.5:
        status_color = 'orange'
        status_text = '⚠️ MODERATE\nMULTICOLLINEARITY'
    else:
        status_color = 'green'
        status_text = '✓ LOW\nMULTICOLLINEARITY'
    
    # Create summary text
    summary_text = f"""
SUMMARY

Status: {status_text}

Dataset:
  • {summary['n_trials']:,} trials
  • {summary['n_sessions']} sessions
  • {summary['n_mice']} mice
  • {summary['n_features']} features

Session-Level:
  • Max VIF: {summary['max_session_vif']:.2f}
  • Max |r|: {summary['max_session_corr']:.3f}
  • High corr pairs: {summary['n_high_corr_pairs']}

Trial-Level:
  • Max VIF: {summary['max_trial_vif']:.2f}
"""
    
    ax5.text(0.05, 0.95, summary_text.strip(), transform=ax5.transAxes,
            verticalalignment='top', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # 6. High correlation pairs (if any)
    ax6 = fig.add_subplot(gs[2, :])
    
    if len(high_corr) > 0:
        # Plot as horizontal bars
        y_pos = np.arange(len(high_corr))
        pair_labels = [f"{row['feature1']}\n<->\n{row['feature2']}" 
                      for _, row in high_corr.iterrows()]
        
        colors_corr = ['darkred' if abs(r) > 0.9 else 'red' if abs(r) > 0.8 else 'orange'
                      for r in high_corr['correlation']]
        
        ax6.barh(y_pos, high_corr['abs_correlation'], color=colors_corr, alpha=0.7,
                edgecolor='black')
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(pair_labels, fontsize=8)
        ax6.set_xlabel('|Correlation| at Session Level', fontsize=11, fontweight='bold')
        ax6.set_title('⚠️ High Correlation Pairs (|r| > 0.7) - Consider Removing One Feature from Each Pair',
                     fontsize=12, fontweight='bold', color='red')
        ax6.axvline(x=0.7, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Threshold (0.7)')
        ax6.axvline(x=0.8, color='red', linestyle='--', linewidth=2, alpha=0.7, label='High (0.8)')
        ax6.axvline(x=0.9, color='darkred', linestyle='--', linewidth=2, alpha=0.7, label='Severe (0.9)')
        ax6.legend(fontsize=8)
        ax6.grid(axis='x', alpha=0.3)
        ax6.set_xlim([0.65, 1.0])
        
        # Add correlation values on bars
        for i, (_, row) in enumerate(high_corr.iterrows()):
            ax6.text(row['abs_correlation'] - 0.02, i, f"{row['correlation']:.3f}",
                    va='center', ha='right', fontweight='bold', fontsize=9, color='white')
    else:
        ax6.text(0.5, 0.5, '✓ No high correlations detected at session level\n\n' +
                'Features are not fundamentally redundant',
                transform=ax6.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3, pad=1))
        ax6.set_xlim([0, 1])
        ax6.set_ylim([0, 1])
        ax6.axis('off')
    
    # Main title with interpretation
    title_color = 'red' if summary['max_session_vif'] > 10 else 'orange' if summary['max_session_vif'] > 5 else 'green'
    plt.suptitle('Multicollinearity Analysis: Trial vs Session Level',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Add recommendation box at bottom
    fig.text(0.5, 0.01, f"RECOMMENDATION: {summary['recommendation']}", 
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=0.5))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    
    # Save if path provided
    if save_path:
        plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}.svg', bbox_inches='tight')
        print(f'Figures saved: {save_path}.png and {save_path}.svg')
    
    return fig


# Example usage
if __name__ == '__main__':
    import pickle
    from pathlib import Path
    
    # Load example data
    experimenter = 'Axel_Bisi'
    dataset_path = Path(f'\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\{experimenter}\\beh_model\\datasets_time_new\\dataset_0')
    data_train = pickle.load(open(dataset_path / 'data_train.pkl', 'rb'))
    
    # Define features to analyze
    features = [
        'whisker', 
        'auditory',
        'time_since_last_auditory_stim',
        'time_since_last_whisker_stim',
        'time_since_last_auditory_reward',
        'time_since_last_whisker_reward'
    ]
    
    # Run analysis
    results = check_multicollinearity(data_train, features)
    
    # Print results
    print_results(results)
    
    # Plot results
    fig = plot_multicollinearity(results, save_path='multicollinearity_analysis')
    plt.show()
    
    # Access specific results
    print('\n\nExample: Accessing specific results')
    print('-'*70)
    print(f"Session-level correlation between whisker and time_since_last_whisker_stim:")
    print(f"  r = {results['session_corr'].loc['whisker', 'time_since_last_whisker_stim']:.3f}")
    
    if len(results['high_corr_pairs']) > 0:
        print("\nFeatures to consider removing:")
        for _, row in results['high_corr_pairs'].iterrows():
            print(f"  - One of: {row['feature1']}, {row['feature2']} (r={row['correlation']:.2f})")
