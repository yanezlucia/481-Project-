import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
import pandas as pd


print("Loading original dataset...")
dataset = load_dataset("HallowsYves/CPSC481-data")
df_original = dataset['train'].to_pandas()

original_features = [col for col in df_original.columns if col != 'Label']

df_processed = pd.read_csv('train_preprocessed.csv')

current_features = [col for col in df_processed.columns if col not in ['Label', 'Binary_Label']]

dropped_features = [f for f in original_features if f not in current_features]

fig, ax = plt.subplots(figsize=(12, 10))

colors = ['#2ecc71' if f in current_features else '#e74c3c' for f in original_features]

y_pos = np.arange(len(original_features))
ax.barh(y_pos, [1]*len(original_features), color=colors)
ax.set_yticks(y_pos)
ax.set_yticklabels(original_features, fontsize=9)
ax.set_xlabel('Feature Status', fontsize=12)
ax.set_title(f'Feature Selection: {len(current_features)}/{len(original_features)} Features Retained', 
             fontsize=14, fontweight='bold')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', label=f'Kept ({len(current_features)})'),
                   Patch(facecolor='#e74c3c', label=f'Removed ({len(dropped_features)})')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('feature_selection_overview.png', dpi=300, bbox_inches='tight')
print(f"\nVisualization saved as 'feature_selection_overview.png'")
print(f"Feature reduction: {len(original_features)} → {len(current_features)} ({((len(original_features) - len(current_features)) / len(original_features)) * 100:.1f}% reduction)")
plt.show()