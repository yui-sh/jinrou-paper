#!/usr/bin/env python3
"""Generate publication-quality figures for the Jinrou paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Output directory
FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# Style settings for NeurIPS
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Color palette
COLORS = {
    'ww': '#c0392b',       # red for werewolf
    'vill': '#2980b9',     # blue for villager
    'accent': '#27ae60',   # green for highlights
    'neutral': '#7f8c8d',  # gray
}

# Model short names (for axis labels)
MODEL_SHORT = [
    'Gemini 3\nPro',
    'Gemini 3\nFlash',
    'GPT-5.2',
    'Claude\nOpus 4.5',
    'Claude\nSonnet 4.5',
    'Grok 4.1\nFast',
    'Grok 4',
    'GPT-5\nmini',
]

MODEL_SHORT_ONELINE = [
    'Gemini 3 Pro',
    'Gemini 3 Flash',
    'GPT-5.2',
    'Claude Opus 4.5',
    'Claude Sonnet 4.5',
    'Grok 4.1 Fast',
    'Grok 4',
    'GPT-5 mini',
]

# ============================================================
# DATA (from REPORT_LANGUAGE.md and FULL_REPORT.md)
# ============================================================

# WW win rates (ordered by performance)
ww_winrate = [44.6, 37.7, 37.8, 36.0, 35.8, 24.4, 22.1, 13.6]

# First-person pronoun usage (%)
pronoun_ww =   [4.40, 3.96, 3.43, 4.36, 4.08, 2.57, 4.09, 4.06]
pronoun_vill = [3.38, 3.22, 2.77, 3.70, 3.22, 2.02, 3.14, 3.40]
pronoun_delta = [w - v for w, v in zip(pronoun_ww, pronoun_vill)]

# Message length (words)
length_ww =   [70.0, 74.5, 100.0, 116.3, 100.9, 43.6, 56.3, 69.9]
length_vill = [66.8, 68.7, 82.8, 115.1, 92.4, 38.8, 49.4, 66.6]
length_delta = [w - v for w, v in zip(length_ww, length_vill)]

# Hedge words per 100 words
hedge_ww =   [0.106, 0.153, 0.176, 0.471, 0.527, 0.123, 0.719, 0.198]
hedge_vill = [0.064, 0.089, 0.183, 0.320, 0.319, 0.078, 0.458, 0.071]
hedge_delta = [w - v for w, v in zip(hedge_ww, hedge_vill)]

# Certainty words per 100 words
cert_ww =   [0.393, 0.389, 0.218, 0.312, 0.317, 0.171, 0.230, 0.108]
cert_vill = [0.492, 0.459, 0.365, 0.440, 0.431, 0.474, 0.409, 0.220]
cert_delta = [w - v for w, v in zip(cert_ww, cert_vill)]

# Sentiment (VADER compound)
sent_ww =   [+0.0814, +0.2493, +0.3062, +0.2485, +0.1471, -0.0656, +0.0336, +0.0799]
sent_vill = [+0.1556, +0.2588, +0.2915, +0.2333, +0.1459, +0.0901, +0.1087, +0.1790]
sent_delta = [w - v for w, v in zip(sent_ww, sent_vill)]

# Deception gap
deception_gap = [-0.037, +0.083, +0.379, +0.120, -0.003, +0.164, +0.317, +0.148]

# Fake Seer claim rate (%)
fake_seer = [7.82, 8.06, 9.97, 9.87, 11.15, 9.16, 12.37, 7.94]

# ============================================================
# FIGURE 1: The Inverted Tell (main finding)
# ============================================================

def fig1_inverted_tell():
    """4-panel figure showing human vs AI deception directions."""
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
    
    features = [
        ('First-Person Pronouns\n(I/me/my)', pronoun_delta, 'Human: Decrease', 'pp'),
        ('Message Length', length_delta, 'Human: Decrease', 'words'),
        ('Hedging Language', hedge_delta, 'Human: Varies', '/100w'),
        ('Negative Sentiment', [-d for d in sent_delta], 'Human: Increase', 'VADER'),
    ]
    
    for ax, (title, deltas, human_dir, unit) in zip(axes.flat, features):
        # Sort by WW win rate order (already sorted)
        x = np.arange(len(MODEL_SHORT))
        colors = [COLORS['ww'] if d > 0 else COLORS['vill'] for d in deltas]
        
        bars = ax.bar(x, deltas, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_SHORT, fontsize=7)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel(f'Δ ({unit})', fontsize=9)
        
        # Add human direction annotation
        ax.annotate(human_dir, xy=(0.98, 0.95), xycoords='axes fraction',
                   fontsize=7, ha='right', va='top', style='italic',
                   color=COLORS['neutral'],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    # Legend
    ww_patch = mpatches.Patch(color=COLORS['ww'], alpha=0.8, label='WW > Villager')
    vill_patch = mpatches.Patch(color=COLORS['vill'], alpha=0.8, label='WW < Villager')
    fig.legend(handles=[ww_patch, vill_patch], loc='lower center', ncol=2, fontsize=9,
              bbox_to_anchor=(0.5, -0.02))
    
    fig.suptitle('The Inverted Tell: AI Deception Reverses Human Patterns', 
                fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig1_inverted_tell.pdf'))
    fig.savefig(os.path.join(FIGDIR, 'fig1_inverted_tell.png'))
    plt.close()
    print("✓ Figure 1: Inverted Tell")


# ============================================================
# FIGURE 2: Best Liars Change Least (scatter)
# ============================================================

def fig2_best_liars():
    """Scatter plot: hedge increase vs WW win rate."""
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))
    
    datasets = [
        ('Hedge Increase\n(per 100 words)', hedge_delta),
        ('Deception Gap\n(sentiment)', deception_gap),
        ('Fake Seer Rate\n(%)', fake_seer),
    ]
    
    for ax, (xlabel, xdata) in zip(axes, datasets):
        ax.scatter(xdata, ww_winrate, c=COLORS['ww'], s=60, alpha=0.8, 
                  edgecolors='white', linewidth=0.5, zorder=5)
        
        # Add model labels
        for i, (x, y) in enumerate(zip(xdata, ww_winrate)):
            name = MODEL_SHORT_ONELINE[i]
            # Offset labels to avoid overlap
            offset_x = 5
            offset_y = 5
            if 'GPT-5 mini' in name:
                offset_y = -10
            elif 'Grok 4.1' in name:
                offset_y = -10
            ax.annotate(name, (x, y), fontsize=6, 
                       xytext=(offset_x, offset_y), textcoords='offset points',
                       color=COLORS['neutral'])
        
        # Trend line
        z = np.polyfit(xdata, ww_winrate, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(xdata), max(xdata), 100)
        ax.plot(x_line, p(x_line), '--', color=COLORS['neutral'], alpha=0.5, linewidth=1)
        
        # Spearman correlation
        from scipy import stats
        rho, pval = stats.spearmanr(xdata, ww_winrate)
        ax.annotate(f'ρ = {rho:.2f}', xy=(0.05, 0.05), xycoords='axes fraction',
                   fontsize=8, fontweight='bold', color=COLORS['ww'])
        
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('WW Win Rate (%)', fontsize=9)
        ax.grid(True, alpha=0.2)
    
    fig.suptitle('The Best Liars Change Least', fontsize=11, fontweight='bold', y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig2_best_liars.pdf'))
    fig.savefig(os.path.join(FIGDIR, 'fig2_best_liars.png'))
    plt.close()
    print("✓ Figure 2: Best Liars Change Least")


# ============================================================
# FIGURE 3: Model Performance Overview
# ============================================================

def fig3_performance():
    """Grouped bar chart: WW vs Villager win rates."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    vill_winrate = [72.0, 71.2, 70.7, 70.4, 67.3, 65.6, 66.2, 64.1]
    
    x = np.arange(len(MODEL_SHORT))
    width = 0.35
    
    bars_ww = ax.bar(x - width/2, ww_winrate, width, label='As Werewolf',
                     color=COLORS['ww'], alpha=0.85, edgecolor='white')
    bars_vill = ax.bar(x + width/2, vill_winrate, width, label='As Villager',
                       color=COLORS['vill'], alpha=0.85, edgecolor='white')
    
    # Add value labels
    for bar in bars_ww:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}',
               ha='center', va='bottom', fontsize=7, color=COLORS['ww'])
    for bar in bars_vill:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}',
               ha='center', va='bottom', fontsize=7, color=COLORS['vill'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_SHORT, fontsize=8)
    ax.set_ylabel('Win Rate (%)', fontsize=10)
    ax.set_title('Model Performance by Role', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 82)
    ax.grid(axis='y', alpha=0.2)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig3_performance.pdf'))
    fig.savefig(os.path.join(FIGDIR, 'fig3_performance.png'))
    plt.close()
    print("✓ Figure 3: Model Performance")


# ============================================================
# FIGURE 4: Pronoun comparison (human vs AI)
# ============================================================

def fig4_pronoun_comparison():
    """Side-by-side: human finding vs AI finding for pronouns."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    
    # Left: Human (schematic from Newman et al.)
    ax1.bar(['Truth', 'Lie'], [4.5, 3.8], color=[COLORS['vill'], COLORS['ww']], alpha=0.8)
    ax1.set_ylabel('First-Person Pronoun %', fontsize=9)
    ax1.set_title('Human Deception\n(Newman et al., 2003)', fontsize=10, fontweight='bold')
    ax1.annotate('↓ Decrease', xy=(0.5, 0.85), xycoords='axes fraction',
                fontsize=10, ha='center', color=COLORS['ww'], fontweight='bold')
    ax1.set_ylim(0, 6)
    
    # Right: AI (our data, averaged)
    avg_vill = np.mean(pronoun_vill)
    avg_ww = np.mean(pronoun_ww)
    ax2.bar(['Villager\n(Truth)', 'Werewolf\n(Lie)'], [avg_vill, avg_ww], 
           color=[COLORS['vill'], COLORS['ww']], alpha=0.8)
    ax2.set_ylabel('First-Person Pronoun %', fontsize=9)
    ax2.set_title('AI Deception\n(This Work, n=8 models)', fontsize=10, fontweight='bold')
    ax2.annotate('↑ Increase', xy=(0.5, 0.85), xycoords='axes fraction',
                fontsize=10, ha='center', color=COLORS['ww'], fontweight='bold')
    ax2.set_ylim(0, 6)
    
    # Add error bars for AI
    ax2.errorbar(['Villager\n(Truth)', 'Werewolf\n(Lie)'], 
                [avg_vill, avg_ww],
                yerr=[np.std(pronoun_vill), np.std(pronoun_ww)],
                fmt='none', color='black', capsize=3, linewidth=1)
    
    fig.suptitle('Pronoun Distancing: Human vs. AI', fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig4_pronoun_comparison.pdf'))
    fig.savefig(os.path.join(FIGDIR, 'fig4_pronoun_comparison.png'))
    plt.close()
    print("✓ Figure 4: Pronoun Comparison")


# ============================================================
# FIGURE 5: Deception Gap heatmap
# ============================================================

def fig5_deception_gap():
    """Horizontal bar chart of deception gap by model."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    # Sort by gap
    models_gap = list(zip(MODEL_SHORT_ONELINE, deception_gap, ww_winrate))
    models_gap.sort(key=lambda x: x[1], reverse=True)
    
    names = [m[0] for m in models_gap]
    gaps = [m[1] for m in models_gap]
    rates = [m[2] for m in models_gap]
    
    colors = [COLORS['ww'] if g > 0 else COLORS['accent'] for g in gaps]
    
    bars = ax.barh(range(len(names)), gaps, color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f'{n} ({r:.1f}%)' for n, r in zip(names, rates)], fontsize=8)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Deception Gap (Public − Private Sentiment)', fontsize=9)
    ax.set_title('The Deception Gap\n(WW win rate in parentheses)', fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.2)
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig5_deception_gap.pdf'))
    fig.savefig(os.path.join(FIGDIR, 'fig5_deception_gap.png'))
    plt.close()
    print("✓ Figure 5: Deception Gap")


# ============================================================
# FIG 6: DECEPTION TYPE DISTRIBUTION (from LLM annotation)
# ============================================================

def fig6_deception_types():
    """Stacked bar chart of deception types by model from LLM annotation."""
    import json
    from collections import Counter, defaultdict
    
    annotation_dir = os.path.join(os.path.dirname(__file__), '..', 'annotation')
    resolved_file = os.path.join(annotation_dir, 'annotations', 'resolved.jsonl')
    sample_file = os.path.join(annotation_dir, 'sample.jsonl')
    
    if not os.path.exists(resolved_file):
        print("⚠ Figure 6: Skipped (no resolved annotations yet)")
        return
    
    # Load sample for model info
    sample = {}
    with open(sample_file) as f:
        for line in f:
            r = json.loads(line)
            sample[r['_sample_id']] = r
    
    # Load resolved annotations
    model_labels = defaultdict(list)
    with open(resolved_file) as f:
        for line in f:
            r = json.loads(line)
            sid = r['sample_id']
            if sid in sample:
                model = sample[sid]['_model']
                model_labels[model].append(r['label'])
    
    if not model_labels:
        print("⚠ Figure 6: Skipped (no resolved annotations)")
        return
    
    LABELS = ['LIE', 'MISDIRECT', 'OMIT', 'TRUTH']
    LABEL_COLORS = {
        'LIE': '#e74c3c',       # red
        'MISDIRECT': '#f39c12', # orange
        'OMIT': '#95a5a6',      # gray
        'TRUTH': '#27ae60',     # green
    }
    
    # Sort models by WW win rate (use MODEL_ORDER if defined, else alphabetical)
    models = sorted(model_labels.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]})
    
    # Left: stacked horizontal bar chart by model
    y_pos = range(len(models))
    bottoms = [0] * len(models)
    
    for label in LABELS:
        widths = []
        for model in models:
            total = len(model_labels[model])
            count = sum(1 for l in model_labels[model] if l == label)
            widths.append(count / total * 100)
        
        ax1.barh(y_pos, widths, left=bottoms, label=label, 
                color=LABEL_COLORS[label], alpha=0.85, edgecolor='white', linewidth=0.5)
        bottoms = [b + w for b, w in zip(bottoms, widths)]
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=8)
    ax1.set_xlabel('Percentage of Werewolf Messages (%)', fontsize=9)
    ax1.set_title('Deception Strategy Distribution by Model', fontsize=10, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax1.set_xlim(0, 100)
    ax1.grid(axis='x', alpha=0.2)
    
    # Right: overall pie chart
    all_labels = []
    for labels in model_labels.values():
        all_labels.extend(labels)
    
    dist = Counter(all_labels)
    sizes = [dist.get(l, 0) for l in LABELS]
    colors = [LABEL_COLORS[l] for l in LABELS]
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=LABELS, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 9})
    for autotext in autotexts:
        autotext.set_fontsize(8)
    ax2.set_title('Overall Distribution\n(n={:,})'.format(len(all_labels)), 
                  fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig6_deception_types.pdf'))
    fig.savefig(os.path.join(FIGDIR, 'fig6_deception_types.png'))
    plt.close()
    print("✓ Figure 6: Deception Types")


# ============================================================
# FIG 7: DECEPTION TYPES BY GAME DAY (temporal dynamics)
# ============================================================

def fig7_deception_temporal():
    """Line chart showing how deception types shift across game days."""
    import json
    from collections import Counter, defaultdict
    
    annotation_dir = os.path.join(os.path.dirname(__file__), '..', 'annotation')
    resolved_file = os.path.join(annotation_dir, 'annotations', 'resolved.jsonl')
    sample_file = os.path.join(annotation_dir, 'sample.jsonl')
    
    if not os.path.exists(resolved_file):
        print("⚠ Figure 7: Skipped (no resolved annotations yet)")
        return
    
    sample = {}
    with open(sample_file) as f:
        for line in f:
            r = json.loads(line)
            sample[r['_sample_id']] = r
    
    day_labels = defaultdict(list)
    with open(resolved_file) as f:
        for line in f:
            r = json.loads(line)
            sid = r['sample_id']
            if sid in sample:
                day = sample[sid]['day']
                day_labels[day].append(r['label'])
    
    if not day_labels:
        print("⚠ Figure 7: Skipped (no resolved annotations)")
        return
    
    LABELS = ['LIE', 'MISDIRECT', 'OMIT', 'TRUTH']
    LABEL_COLORS = {
        'LIE': '#e74c3c',
        'MISDIRECT': '#f39c12',
        'OMIT': '#95a5a6',
        'TRUTH': '#27ae60',
    }
    
    days = sorted(d for d in day_labels.keys() if len(day_labels[d]) >= 20)  # min 20 samples
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for label in LABELS:
        pcts = []
        for day in days:
            total = len(day_labels[day])
            count = sum(1 for l in day_labels[day] if l == label)
            pcts.append(count / total * 100)
        
        ax.plot(days, pcts, marker='o', label=label, color=LABEL_COLORS[label], linewidth=2)
    
    ax.set_xlabel('Game Day', fontsize=10)
    ax.set_ylabel('Percentage of Messages (%)', fontsize=10)
    ax.set_title('Deception Strategy Evolution Over Game Days', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_xticks(days)
    ax.set_xticklabels([f'Day {d}' for d in days])
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig7_deception_temporal.pdf'))
    fig.savefig(os.path.join(FIGDIR, 'fig7_deception_temporal.png'))
    plt.close()
    print("✓ Figure 7: Deception Temporal Dynamics")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    fig1_inverted_tell()
    fig2_best_liars()
    fig3_performance()
    fig4_pronoun_comparison()
    fig5_deception_gap()
    fig6_deception_types()
    fig7_deception_temporal()
    print(f"\nAll figures saved to {FIGDIR}/")
