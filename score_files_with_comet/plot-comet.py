import matplotlib.pyplot as plt
import numpy as np

# Read data from the file
filename = 'en-de/cc-matrix/cc-matrix-ende-cometkiwi.tsv'  # Replace with the actual file name
#filename = 'en-de//europarl/europarl-v10.de-en-cometkiwi.tsv'
#filename = 'en-de/news-commentary/news-commentary-v18.de-en-cometkiwi.tsv'

# Assuming your file has three columns and you want to extract the score from the lines
with open(filename, 'r') as file:
    #scores = [float(line.split()[-2]) for line in file if float(line.split()[-1]) < 0.75]
    scores = [float(line.split()[-2]) for line in file]

# Create histogram bins
bins = np.arange(0.0, 1.0, 0.02)

# Plot histogram
plt.hist(scores, bins=bins, edgecolor='black', alpha=0.7)
plt.title('Distribution of Scores')
plt.xlabel('Score Bins')
plt.ylabel('Number of Lines')
plt.xticks(bins)
plt.grid(True)
plt.show()

