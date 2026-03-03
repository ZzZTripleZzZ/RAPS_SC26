import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Reading the CSV file
csv_file_path = 'app1.csv'  # Replace with the correct path to your CSV file
df = pd.read_csv(csv_file_path)

# Extracting the rank names
ranks = df['component'].str.extract('(\d+)')[0].astype(int)

# Extracting the spy data
spy_data = df.filter(regex='spy\d+')

# Plotting the spyplot
plt.figure(figsize=(12, 8))
plt.imshow(spy_data, aspect='auto', interpolation='none')
plt.colorbar(label='Bytes')
plt.xticks(range(spy_data.shape[1]), spy_data.columns, rotation=45)
plt.yticks(range(len(ranks)), ranks)
plt.xlabel('Spy Indices')
plt.ylabel('Ranks')
plt.title('Spyplot for SST/macro Output')
plt.axis('off') 

# Save the plot as a PNG file
plt.savefig('spyplot.png', format='png')
plt.close()
