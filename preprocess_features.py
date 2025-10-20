import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/features/extracted_features.csv')

# Fill NaN in perplexity with mean
df['perplexity'] = df['perplexity'].fillna(df['perplexity'].mean())

# Cap perplexity at 1000
df['perplexity'] = np.clip(df['perplexity'], None, 1000)

# Log-transform perplexity
df['perplexity'] = np.log(df['perplexity'] + 1e-8)

# StandardScaler
scaler = StandardScaler()
df['perplexity'] = scaler.fit_transform(df[['perplexity']])

# Save to new file
df.to_csv('data/features/extracted_features_processed.csv', index=False)

print("Processed perplexity saved to data/features/extracted_features_processed.csv")