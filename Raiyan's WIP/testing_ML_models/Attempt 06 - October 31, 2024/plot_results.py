import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import os

# Load data from metrics files
other_models_metrics_path = 'Results/metrics_other_models.csv'
xgboost_metrics_path = 'Results/metrics_xgboost.csv'

# Check if files exist before loading
if not os.path.exists(other_models_metrics_path) or not os.path.exists(xgboost_metrics_path):
    raise FileNotFoundError("Ensure that 'metrics_other_models.csv' and 'metrics_xgboost.csv' exist in the 'Results/' directory.")

# Load data into DataFrames
other_models_df = pd.read_csv(other_models_metrics_path)
xgboost_df = pd.read_csv(xgboost_metrics_path)

# Add a 'Model' column to each DataFrame for consistency
other_models_df['Model'] = other_models_df['Model'].str.replace('_', ' ').str.title()
xgboost_df['Model'] = 'XGBoost'

# Merge the data into a single DataFrame for easier plotting
metrics_df = pd.concat([other_models_df, xgboost_df], ignore_index=True)

# List of metrics to plot
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity']

# Set up color palette for the models
unique_models = metrics_df['Model'].unique()
palette = sns.color_palette("husl", len(unique_models))  # Color palette with unique colors for each model
model_colors = dict(zip(unique_models, palette))

# Create subplots for each metric in a single figure
fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 6), sharey=True)

for i, metric in enumerate(metrics_to_plot):
    sns.barplot(
        data=metrics_df,
        x='Model',
        y=metric,
        palette=model_colors,
        ax=axes[i]
    )
    axes[i].set_title(f'{metric}')
    axes[i].set_xlabel('Model')
    axes[i].set_ylabel(f'{metric} (%)' if i == 0 else '')
    axes[i].tick_params(axis='x', rotation=45)

# Add a legend to the figure
handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='', markersize=10) for color in model_colors.values()]
labels = list(model_colors.keys())
fig.legend(handles, labels, loc='upper center', ncol=len(labels), title='Models')

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle('Model Performance Metrics Comparison')
plt.savefig('Results/metrics_comparison.png', bbox_inches='tight', dpi=300)
print("Saved metrics comparison plot as 'Results/metrics_comparison.png'.")

# Extract and prepare the best hyperparameters table for display
hyperparameters_table = metrics_df[['Model', 'Best Parameters']].drop_duplicates().reset_index(drop=True)

# Render and save the table as an image using Matplotlib
fig, ax = plt.subplots(figsize=(10, len(hyperparameters_table) * 0.6))  # Dynamic height based on table rows
ax.axis('off')
table = ax.table(
    cellText=hyperparameters_table.values,
    colLabels=hyperparameters_table.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width([0, 1])

# Save the table as an image
table_image_path = 'Results/optimal_hyperparameters.png'
plt.savefig(table_image_path, bbox_inches='tight', dpi=300)
print(f"Saved optimal hyperparameters table as '{table_image_path}'.")
