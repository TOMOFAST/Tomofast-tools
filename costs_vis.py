import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

# ---------------------------
# Load file & column names
# ---------------------------
file_path = "..your_folder\\costs.txt"  

file_path = Path(file_path)

with open(file_path, "r") as f:
    header_line = f.readline().strip()
columns = [re.split(r":", item)[1] for item in header_line.replace("#", "").split(",")]
df = pd.read_csv(file_path, sep=r"\s+", header=None, skiprows=1, on_bad_lines='skip')
df.columns = columns
df = df.apply(pd.to_numeric, errors="coerce")

# Drop rows with NaN (truncated lines)
df = df.dropna()
it = df["iteration"]

print(f"Loaded {len(df)} complete iterations (0 to {int(it.max())})")

# ---------------------------
# Detect active data types
# ---------------------------
def has_data(col_name):
    """Check if column exists and has non-zero values"""
    return col_name in df.columns and df[col_name].abs().max() > 0

has_grav = has_data('data_cost_grav')
has_mag = has_data('data_cost_mag')

# Determine inversion type
if has_grav and has_mag:
    inversion_type = "Joint Gravity-Magnetic"
    data_types = ['grav', 'mag']
    data_labels = {'grav': 'Gravity', 'mag': 'Magnetic'}
    data_colors = {'grav': '#E63946', 'mag': '#457B9D'}
elif has_grav:
    inversion_type = "Gravity"
    data_types = ['grav']
    data_labels = {'grav': 'Gravity'}
    data_colors = {'grav': '#E63946'}
elif has_mag:
    inversion_type = "Magnetic"
    data_types = ['mag']
    data_labels = {'mag': 'Magnetic'}
    data_colors = {'mag': '#457B9D'}
else:
    raise ValueError("No gravity or magnetic data found in costs file!")

print(f"Detected inversion type: {inversion_type}")

# ---------------------------
# Build figure
# ---------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)
axes = axes.ravel()

colors_xyz = ['#E63946', '#2A9D8F', '#264653']

# ---------------------------
# 1 — Data misfit
# ---------------------------
ax = axes[0]
for dtype in data_types:
    col = f'data_cost_{dtype}'
    ax.plot(it, df[col], 'o-', linewidth=1.5, markersize=3, 
            color=data_colors[dtype], label=data_labels[dtype])
    # Annotate final value
    final_val = df[col].iloc[-1]
    ax.annotate(f'{final_val:.2e}', xy=(it.iloc[-1], final_val), 
                xytext=(5, 5), textcoords='offset points', fontsize=9, color=data_colors[dtype])
ax.set_title("Data Misfit", fontsize=12, fontweight='bold')
ax.set_yscale('log')
if len(data_types) > 1:
    ax.legend(fontsize=9)

# ---------------------------
# 2 — Model regularization
# ---------------------------
ax = axes[1]
for dtype in data_types:
    col = f'model_cost_{dtype}'
    if has_data(col):
        ax.plot(it, df[col], 'o-', linewidth=1.5, markersize=3, 
                color=data_colors[dtype], label=data_labels[dtype])
ax.set_title("Model Damping Cost", fontsize=12, fontweight='bold')
ax.set_yscale('log')
if len(data_types) > 1:
    ax.legend(fontsize=9)

# ---------------------------
# 3 — ADMM cost
# ---------------------------
ax = axes[2]
for dtype in data_types:
    col = f'ADMM_cost_{dtype}'
    if has_data(col):
        ax.plot(it, df[col], 'o-', linewidth=1.5, markersize=3, 
                color=data_colors[dtype], label=data_labels[dtype])
ax.set_title("ADMM Cost", fontsize=12, fontweight='bold')
ax.set_yscale('log')
if len(data_types) > 1:
    ax.legend(fontsize=9)

# ---------------------------
# 4 — Gradient damping (x, y, z)
# ---------------------------
ax = axes[3]
for dtype in data_types:
    linestyle = '-' if dtype == 'grav' else '--'
    for i, direction in enumerate(['x', 'y', 'z']):
        col = f'damp_gradient_cost_{direction}_{dtype}'
        if has_data(col):
            label = f'∇{direction}' if len(data_types) == 1 else f'∇{direction} ({data_labels[dtype]})'
            ax.plot(it, df[col], linestyle, marker='o', linewidth=1.5, markersize=2, 
                    color=colors_xyz[i], label=label, alpha=0.8 if dtype == 'grav' else 0.6)
ax.set_title("Gradient Damping", fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=8, ncol=2 if len(data_types) > 1 else 1)

# ---------------------------
# 5 — Total cost breakdown
# ---------------------------
ax = axes[4]

# Calculate totals for each component
df['total_cost'] = 0
component_totals = {}

# Data cost
component_totals['Data'] = sum(df[f'data_cost_{dtype}'] for dtype in data_types if has_data(f'data_cost_{dtype}'))
df['total_cost'] += component_totals['Data']

# Model cost
model_cols = [f'model_cost_{dtype}' for dtype in data_types if has_data(f'model_cost_{dtype}')]
if model_cols:
    component_totals['Model'] = sum(df[col] for col in model_cols)
    df['total_cost'] += component_totals['Model']

# ADMM cost
admm_cols = [f'ADMM_cost_{dtype}' for dtype in data_types if has_data(f'ADMM_cost_{dtype}')]
if admm_cols:
    component_totals['ADMM'] = sum(df[col] for col in admm_cols)
    df['total_cost'] += component_totals['ADMM']

# Gradient cost
gradient_sum = 0
for dtype in data_types:
    for direction in ['x', 'y', 'z']:
        col = f'damp_gradient_cost_{direction}_{dtype}'
        if has_data(col):
            gradient_sum += df[col]
if isinstance(gradient_sum, pd.Series) and gradient_sum.abs().max() > 0:
    component_totals['Gradient'] = gradient_sum
    df['total_cost'] += gradient_sum

# Cross-gradient (for joint inversion)
cross_grad_sum = 0
for direction in ['x', 'y', 'z']:
    col = f'cross_grad_cost_{direction}'
    if has_data(col):
        cross_grad_sum += df[col]
if isinstance(cross_grad_sum, pd.Series) and cross_grad_sum.abs().max() > 0:
    component_totals['Cross-grad'] = cross_grad_sum
    df['total_cost'] += cross_grad_sum

# Clustering cost
clustering_sum = 0
for dtype in data_types:
    col = f'clustering_cost_{dtype}'
    if has_data(col):
        clustering_sum += df[col]
if isinstance(clustering_sum, pd.Series) and clustering_sum.abs().max() > 0:
    component_totals['Clustering'] = clustering_sum
    df['total_cost'] += clustering_sum

# Plot components
component_colors = {
    'Data': '#E63946', 'Model': '#457B9D', 'ADMM': '#2A9D8F', 
    'Gradient': '#F4A261', 'Cross-grad': '#9B5DE5', 'Clustering': '#00BBF9'
}
for name, values in component_totals.items():
    if isinstance(values, pd.Series) and values.abs().max() > 0:
        ax.plot(it, values, '-', linewidth=2, label=name, color=component_colors.get(name, 'gray'))

ax.plot(it, df['total_cost'], 'k--', linewidth=2, label='Total', alpha=0.7)
ax.set_title("Cost Components Comparison", fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=9, ncol=2)

# ---------------------------
# 6 — Convergence: relative change per iteration
# ---------------------------
ax = axes[5]
relative_change = np.abs(df['total_cost'].diff()) / df['total_cost'].shift(1) * 100
ax.plot(it[1:], relative_change[1:], 'ko-', linewidth=1.5, markersize=3)
ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=1, label='1% threshold')
ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, linewidth=1, label='0.1% threshold')
ax.set_title("Convergence: Relative Change (%)", fontsize=12, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=9)
ax.set_ylim(bottom=0.001)

# ---------------------------
# Enhanced grids for all axes
# ---------------------------
for ax in axes:
    ax.minorticks_on()
    ax.grid(True, which='major', color='gray', alpha=0.5, linestyle='-', linewidth=0.7)
    ax.grid(True, which='minor', color='gray', alpha=0.3, linestyle=':', linewidth=0.5)

# ---------------------------
# Axis labels
# ---------------------------
for ax in axes[-3:]:
    ax.set_xlabel("Iteration", fontsize=11)
for ax in axes:
    ax.set_ylabel("Cost", fontsize=10)

# ---------------------------
# Title and layout
# ---------------------------
fig.suptitle(f"{inversion_type} Inversion Cost Evolution", fontsize=16, fontweight='bold')
fig.tight_layout(rect=[0, 0.05, 1, 0.96])

# ---------------------------
# Summary statistics box
# ---------------------------
initial_cost = df['total_cost'].iloc[0]
final_cost = df['total_cost'].iloc[-1]
reduction = (1 - final_cost/initial_cost) * 100
n_iter = int(it.max())
final_change = relative_change.iloc[-1]

summary = (f"Type: {inversion_type}\n"
           f"Iterations: {n_iter}\n"
           f"Initial cost: {initial_cost:.2e}\n"
           f"Final cost: {final_cost:.2e}\n"
           f"Reduction: {reduction:.1f}%\n"
           f"Final Δ: {final_change:.3f}%")

fig.text(0.01, 0.01, summary, fontsize=9, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='gray'))

# ---------------------------
# Save
# ---------------------------
plt.savefig("cost_evolution.png", dpi=200, bbox_inches="tight", facecolor='white')
plt.show()

# ---------------------------
# Print summary
# ---------------------------
print("\n" + "="*50)
print("CONVERGENCE SUMMARY")
print("="*50)
print(f"Inversion type: {inversion_type}")
print(f"Total iterations: {n_iter}")
print(f"Initial total cost: {initial_cost:.4e}")
print(f"Final total cost: {final_cost:.4e}")
print(f"Cost reduction: {reduction:.2f}%")
print(f"Final relative change: {final_change:.4f}%")
print(f"Converged (< 0.1% change): {'Yes ✓' if final_change < 0.1 else 'No - consider more iterations'}")
