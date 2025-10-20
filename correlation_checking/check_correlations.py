import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_correlation_pairs(corr_matrix):
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            pairs.append({
                'feature_1': corr_matrix.columns[i],
                'feature_2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i, j],
                'abs_correlation': abs(corr_matrix.iloc[i, j])
            })
    
    pairs_df = pd.DataFrame(pairs).sort_values('abs_correlation', ascending=False)
    return pairs_df


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features_path = os.path.join(script_dir,"../", "data", "features", "extracted_features.csv")
    
    features_df = pd.read_csv(features_path)
    
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    numeric_df = features_df[numeric_cols]
    if 'generated' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['generated'])
    
    corr_matrix = numeric_df.corr()
    
    pairs_df = get_correlation_pairs(corr_matrix)
    
    output_csv = os.path.join(script_dir, "feature_correlations_sorted.csv")
    pairs_df.to_csv(output_csv, index=False)
    
    plt.figure(figsize=(20, 18))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                cmap='coolwarm', 
                center=0, 
                vmin=-1, 
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Correlation"},
                annot=False)
    
    plt.title('Feature Correlation Matrix', fontsize=20, pad=20, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    output_png = os.path.join(script_dir, "correlation_heatmap.png")
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    main()
