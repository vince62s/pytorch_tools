# Function to read scores from the comet file
import os.path
def read_scores(filename):
    source = []
    target = []
    fwd = []
    bwd = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            fields = line.strip().split('\t')
            if len(fields) == 3:
                source.append(fields[0])
                target.append(fields[1])
                fwd.append(float(fields[2]))
            else:
                print(fields)
    return source, target, fwd

# Set the threshold
threshold1 = 0.73  # Adjust as needed
cap1 = 1.00

folder='en-fr/bt_fromfr/news2019-fromfr/'
filename='news.2019.fr.shuffled.btfromfr.en-fr.cometkiwi'
# Read scores from the comet file
source, target, scores_fwd = read_scores(folder+filename+'.tsv')

filtered_data_2 = [tgt + '\n' for src, tgt, fwd in zip(source, target, scores_fwd) if (fwd >= threshold1) and (fwd <= cap1)]
filtered_data_1 = [src + '\n' for src, tgt, fwd in zip(source, target, scores_fwd) if (fwd >= threshold1) and (fwd <= cap1)]

with open(folder+'filtered_' + os.path.basename(folder+filename) + '.fr', 'w') as file:
    file.writelines(filtered_data_2)
with open(folder+'filtered_' + os.path.basename(folder+filename) + '.en', 'w') as file:
    file.writelines(filtered_data_1)


