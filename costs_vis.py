'''
A script for visualisation of Tomofast-x inversion cost evolution.

Author: Jeremie Giraud

Note:   Visualisation for the cost of the 3 components of the cross-gradient separately is not implemented. 
        It is currently only for the sum of all three. 
'''

import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
import re
from pathlib import Path

#==================================================================================================
# Check if column has non-zero data.
def has_data(df, col_name):
    '''
    Check if column exists and has non-zero values.
    '''
    return col_name in df.columns and df[col_name].abs().max() > 0

#==================================================================================================
# Detect inversion type from cost data.
def detect_inversion_type(df):
    '''
    Detect whether this is gravity, magnetic, or joint inversion.
    
    Returns:
        inversion_type: str
        data_types: list of active data types ['grav', 'mag']
        data_labels: dict mapping type to label
        data_colors: dict mapping type to color
    '''
    has_grav = has_data(df, 'data_cost_grav')
    has_mag = has_data(df, 'data_cost_mag')

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

    return inversion_type, data_types, data_labels, data_colors

#==================================================================================================
# Calculate total cost from all components.
def calculate_costs(df, data_types):
    '''
    Calculate total cost and component totals.
    
    Returns:
        df: DataFrame with total_cost column added
        component_totals: dict of component name to Series
    '''
    component_totals = {}

    #----------------------------------------------------------------------------------
    # Data cost.
    #----------------------------------------------------------------------------------
    data_sum = sum(df[f'data_cost_{dtype}'] for dtype in data_types if has_data(df, f'data_cost_{dtype}'))
    if isinstance(data_sum, pd.Series):
        component_totals['Data'] = data_sum

    #----------------------------------------------------------------------------------
    # Model cost.
    #----------------------------------------------------------------------------------
    model_cols = [f'model_cost_{dtype}' for dtype in data_types if has_data(df, f'model_cost_{dtype}')]
    if model_cols:
        model_sum = sum(df[col] for col in model_cols)
        component_totals['Model'] = model_sum

    #----------------------------------------------------------------------------------
    # ADMM cost.
    #----------------------------------------------------------------------------------
    admm_cols = [f'ADMM_cost_{dtype}' for dtype in data_types if has_data(df, f'ADMM_cost_{dtype}')]
    if admm_cols:
        admm_sum = sum(df[col] for col in admm_cols)
        component_totals['ADMM'] = admm_sum

    #----------------------------------------------------------------------------------
    # Gradient damping cost.
    #----------------------------------------------------------------------------------
    gradient_sum = 0
    for dtype in data_types:
        for direction in ['x', 'y', 'z']:
            col = f'damp_gradient_cost_{direction}_{dtype}'
            if has_data(df, col):
                gradient_sum += df[col]
    if isinstance(gradient_sum, pd.Series) and gradient_sum.abs().max() > 0:
        component_totals['Gradient'] = gradient_sum

    #----------------------------------------------------------------------------------
    # Cross-gradient cost.
    #----------------------------------------------------------------------------------
    cross_grad_sum = 0
    for direction in ['x', 'y', 'z']:
        col = f'cross_grad_cost_{direction}'
        if has_data(df, col):
            cross_grad_sum += df[col]
    if isinstance(cross_grad_sum, pd.Series) and cross_grad_sum.abs().max() > 0:
        component_totals['Cross-grad'] = cross_grad_sum

    #----------------------------------------------------------------------------------
    # Clustering cost.
    #----------------------------------------------------------------------------------
    clustering_sum = 0
    for dtype in data_types:
        col = f'clustering_cost_{dtype}'
        if has_data(df, col):
            clustering_sum += df[col]
    if isinstance(clustering_sum, pd.Series) and clustering_sum.abs().max() > 0:
        component_totals['Clustering'] = clustering_sum

    return df, component_totals

#==================================================================================================
# Apply enhanced grid to axes.
def apply_grid(ax):
    '''
    Apply major and minor grid lines to axis.
    '''
    ax.minorticks_on()
    ax.grid(True, which='major', color='gray', alpha=0.5, linestyle='-', linewidth=0.7)
    ax.grid(True, which='minor', color='gray', alpha=0.3, linestyle=':', linewidth=0.5)

#==================================================================================================
# Draw data misfit panel.
def draw_data_misfit(ax, df, it, data_types, data_labels, data_colors):
    '''
    Draw the data misfit cost evolution.
    '''
    for dtype in data_types:
        col = f'data_cost_{dtype}'
        ax.plot(it, df[col], 'o-', linewidth=1.5, markersize=3,
                color=data_colors[dtype], label=data_labels[dtype])
        # Annotate final value.
        final_val = df[col].iloc[-1]
        ax.annotate(f'{final_val:.2e}', xy=(it.iloc[-1], final_val),
                    xytext=(5, 5), textcoords='offset points', fontsize=9, color=data_colors[dtype])

    ax.set_title("Data Misfit", fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    if len(data_types) > 1:
        ax.legend(fontsize=9)

#==================================================================================================
# Draw model regularization panel.
def draw_model_cost(ax, df, it, data_types, data_labels, data_colors):
    '''
    Draw the model regularization cost evolution.
    '''
    for dtype in data_types:
        col = f'model_cost_{dtype}'
        if has_data(df, col):
            ax.plot(it, df[col], 'o-', linewidth=1.5, markersize=3,
                    color=data_colors[dtype], label=data_labels[dtype])

    ax.set_title("Model Regularization", fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    if len(data_types) > 1:
        ax.legend(fontsize=9)

#==================================================================================================
# Draw ADMM cost panel.
def draw_admm_cost(ax, df, it, data_types, data_labels, data_colors):
    '''
    Draw the ADMM cost evolution.
    '''
    for dtype in data_types:
        col = f'ADMM_cost_{dtype}'
        if has_data(df, col):
            ax.plot(it, df[col], 'o-', linewidth=1.5, markersize=3,
                    color=data_colors[dtype], label=data_labels[dtype])

    ax.set_title("ADMM Cost", fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    if len(data_types) > 1:
        ax.legend(fontsize=9)

#==================================================================================================
# Draw gradient damping panel.
def draw_gradient_damping(ax, df, it, data_types, data_labels):
    '''
    Draw the gradient damping cost evolution (x, y, z components).
    '''
    colors_xyz = ['#E63946', '#2A9D8F', '#264653']

    for dtype in data_types:
        linestyle = '-' if dtype == 'grav' else '--'
        for i, direction in enumerate(['x', 'y', 'z']):
            col = f'damp_gradient_cost_{direction}_{dtype}'
            if has_data(df, col):
                if len(data_types) == 1:
                    label = f'∇{direction}'
                else:
                    label = f'∇{direction} ({data_labels[dtype]})'
                ax.plot(it, df[col], linestyle, marker='o', linewidth=1.5, markersize=2,
                        color=colors_xyz[i], label=label, alpha=0.8 if dtype == 'grav' else 0.6)

    ax.set_title("Gradient Damping", fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=8, ncol=2 if len(data_types) > 1 else 1)

#==================================================================================================
# Draw cross-gradient or clustering panel.
def draw_clust_or_xgrad_cost(ax, it, component_totals):
    """
    Draw Cross-gradient or Clustering cost evolution if present.
    """
    plotted = False

    if 'Cross-grad' in component_totals:
        ax.plot(it, component_totals['Cross-grad'],
                'o-', linewidth=1.5, markersize=3,
                color='#9B5DE5', label='Cross-gradient')
        plotted = True

    if 'Clustering' in component_totals:
        ax.plot(it, component_totals['Clustering'],
                's-', linewidth=1.5, markersize=3,
                color='#00BBF9', label='Clustering')
        plotted = True

    if plotted:
        ax.set_title("Additional Regularization Terms", fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
    else:
        ax.axis("off")  # Hide if nothing to show

#==================================================================================================
# Draw cost components comparison panel.
def draw_cost_components(ax, it, component_totals):
    '''
    Draw comparison of all cost components.
    '''
    component_colors = {
        'Data': '#E63946',
        'Model': '#457B9D',
        'ADMM': '#2A9D8F',
        'Gradient': '#F4A261',
        'Cross-grad': '#9B5DE5',
        'Clustering': '#00BBF9'
    }

    for name, values in component_totals.items():
        if isinstance(values, pd.Series) and values.abs().max() > 0:
            ax.plot(it, values, '-', linewidth=2, label=name,
                    color=component_colors.get(name, 'gray'))

    ax.set_title("Cost Components Comparison (no weights from cost function)", fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=9, ncol=2)

#==================================================================================================
# Main function for cost evolution visualisation.
def main(filename_costs, filename_output='cost_evolution.png', save_figure=False):
    '''
    Main function to visualise inversion cost evolution.

    Parameters:
        filename_costs: Path to the costs.txt file from Tomofast-x output.
        filename_output: Path for saving the output figure.
        save_figure: bool to determine whether figure will be saved or not.
    '''
    print('Started cost_evolution_vis.')

    #----------------------------------------------------------------------------------
    # Reading data.
    #----------------------------------------------------------------------------------
    file_path = Path(filename_costs)
    with open(file_path, "r") as f:
        header_line = f.readline().strip()

    columns = [re.split(r":", item)[1] for item in header_line.replace("#", "").split(",")]
    df = pd.read_csv(file_path, sep=r"\s+", header=None, skiprows=1, on_bad_lines='skip')
    df.columns = columns
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with NaN (truncated lines).
    df = df.dropna()
    it = df["iteration"]

    print(f"Loaded {len(df)} complete iterations (0 to {int(it.max())})")

    #----------------------------------------------------------------------------------
    # Detect inversion type.
    #----------------------------------------------------------------------------------
    inversion_type, data_types, data_labels, data_colors = detect_inversion_type(df)
    print(f"Detected inversion type: {inversion_type}")

    #----------------------------------------------------------------------------------
    # Calculate total cost.
    #----------------------------------------------------------------------------------
    df, component_totals = calculate_costs(df, data_types)

    #----------------------------------------------------------------------------------
    # Create figure.
    #----------------------------------------------------------------------------------
    fig, axes = pl.subplots(2, 3, figsize=(16, 10), sharex=True)
    axes = axes.ravel()

    # Panel 5 not used. 
    axes[5].axis("off")

    #----------------------------------------------------------------------------------
    # Draw panels.
    #----------------------------------------------------------------------------------
    draw_data_misfit(axes[0], df, it, data_types, data_labels, data_colors)
    draw_model_cost(axes[1], df, it, data_types, data_labels, data_colors)
    draw_admm_cost(axes[2], df, it, data_types, data_labels, data_colors)
    draw_gradient_damping(axes[3], df, it, data_types, data_labels)
    draw_cost_components(axes[4], it, component_totals)
    draw_clust_or_xgrad_cost(axes[5], it, component_totals)

    #----------------------------------------------------------------------------------
    # Apply grids to all axes.
    #----------------------------------------------------------------------------------
    for ax in axes[:-1]:
        apply_grid(ax)

    #----------------------------------------------------------------------------------
    # Axis labels.
    #----------------------------------------------------------------------------------
    for ax in axes[-3:]:
        ax.set_xlabel("Iteration", fontsize=11)
    for ax in axes:
        ax.set_ylabel("Cost", fontsize=10)

    #----------------------------------------------------------------------------------
    # Title and layout.
    #----------------------------------------------------------------------------------
    fig.suptitle(f"{inversion_type} Inversion Cost Evolution", fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    #----------------------------------------------------------------------------------
    # Save figure.
    #----------------------------------------------------------------------------------
    if save_figure:
        pl.savefig(filename_output, dpi=200, bbox_inches="tight", facecolor='white')
        print(f"Figure saved to: {filename_output}")

    pl.show()
    pl.close(pl.gcf())


#=============================================================================
if __name__ == "__main__":

    # Example usage for Tomofast-x output.

    # Path to costs file from Tomofast-x output.
    filename_costs = "./outputs/costs.txt"

    # Path for output figure.
    filename_output = './output/cost_evolution.png'

    main(filename_costs, filename_output, save_figure=True)
