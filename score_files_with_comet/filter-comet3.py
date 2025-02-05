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
            source.append(fields[0])
            target.append(fields[1])
            fwd.append(float(fields[2]))
            bwd.append(float(fields[3]))
    return source, target, fwd, bwd

# Set the threshold
threshold1 = 0.82  # Adjust as needed
threshold2 = 0.82  # Adjust as needed
cap1 = 1.00
cap2 = 1.00

# Read scores from the comet file
#source, target, scores_fwd, scores_bwd = read_scores('paracrawl/paracrawlv9.en-de-cometkiwi.tsv')
source, target, scores_fwd, scores_bwd = read_scores('cc-matrix/cc-matrix-ende-cometkiwi.tsv')

#filtered_data_en = [src + '\n' for src, tgt, fwd, bwd in zip(source, target, scores_fwd, scores_bwd) if (fwd > threshold1) and (fwd < cap1) and (bwd > threshold2) and (bwd < cap2)]
filtered_data_en = [src + '\n' for src, tgt, fwd, bwd in zip(source, target, scores_fwd, scores_bwd) if (fwd > threshold1) and (fwd < cap1)]

#filtered_data_de = [tgt + '\n' for src, tgt, fwd, bwd in zip(source, target, scores_fwd, scores_bwd) if (fwd > threshold1) and (fwd < cap1) and (bwd > threshold2) and (bwd < cap2)]
filtered_data_de = [tgt + '\n' for src, tgt, fwd, bwd in zip(source, target, scores_fwd, scores_bwd) if (fwd > threshold1) and (fwd < cap1)]

#with open('paracrawl/filtered_' + os.path.basename('paracrawl/paracrawlv9.en-de-comet820') + '.de', 'w') as file:
with open('cc-matrix/filtered_' + os.path.basename('cc-matrix/cc-matrix-ende-orig-en-comet820') + '.de', 'w') as file:
    file.writelines(filtered_data_de)

#with open('paracrawl/filtered_' + os.path.basename('paracrawl/paracrawlv9.en-de-comet820') + '.en', 'w') as file:
with open('cc-matrix/filtered_' + os.path.basename('cc-matrix/cc-matrix-ende-orig-en-comet820') + '.en', 'w') as file:
    file.writelines(filtered_data_en)


