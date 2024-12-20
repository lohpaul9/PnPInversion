import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file into a DataFrame
file_path = './evaluation_result.csv'  # Replace with your CSV file path
res_df = pd.read_csv(file_path)

avg_metrics = res_df.mean(axis=0).to_numpy()[1:]
np_df = avg_metrics.reshape(len(avg_metrics) // 5, 5)

df = pd.DataFrame(np_df, columns=['LPIPS', 'LPIPS (unedit)', 'SSIM', 'CLIP', 'CLIP (edit)'])
df.insert(0, "Model", ["DDIM+P2P", "DDIM+MasaCtrl", "DirInv+PNP", "Sketchy"])

# Plot the table
fig, ax = plt.subplots(figsize=(len(df.columns) * 2, len(df) * 0.5))  # Adjust size as needed
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

# Display the table
print(df)
plt.savefig('eval_table_final.png')
plt.show()