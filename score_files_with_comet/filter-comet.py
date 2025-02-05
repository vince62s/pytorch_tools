# Function to read scores from the comet file
import os.path
def read_scores(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        scores = [float(line.split()[-1]) for line in lines]
    return scores

# Set the threshold
threshold = 0.725  # Adjust as needed

# Read scores from the comet file
scores_fwd = read_scores('en-de/bt_fromde/wikipedia/comet-wikipedia.txt')

file_de = 'en-de/bt_fromde/wikipedia/wikipedia-de.txt.clean.scored.de.13.de'
filtered_data_de = [line for line, score_fwd in zip(open(file_de, 'r'), scores_fwd) if (score_fwd > threshold)]

file_en = 'en-de/bt_fromde/wikipedia/wikipedia-de.txt.clean.scored.de.13.en'
filtered_data_en = [line for line, score_fwd in zip(open(file_en, 'r'), scores_fwd) if (score_fwd > threshold)]

# Save the filtered data to new files if needed
with open('en-de/bt_fromde/wikipedia/filtered_' + os.path.basename(file_de), 'w') as file:
    file.writelines(filtered_data_de)

with open('en-de/bt_fromde/wikipedia/filtered_' + os.path.basename(file_en), 'w') as file:
    file.writelines(filtered_data_en)


