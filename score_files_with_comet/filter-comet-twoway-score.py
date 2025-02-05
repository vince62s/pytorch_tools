# Function to read scores from the comet file
import os.path
def read_scores(filename):
    source = []
    target = []
    fwd = []
    bwd = []
    with open(filename, 'rb') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            try:
                decode_line = line.decode('utf-8')
            except UnicodeDecodeError:
                print(line)
                raise UnicodeDecodeError('wrong encoding')
            fields = decode_line.strip().split('\t')
            if len(fields) == 4:
                source.append(fields[0])
                target.append(fields[1])
                fwd.append(float(fields[2]))
                bwd.append(float(fields[3]))
            else:
                if fields == ['0.4612', '0.4612']:
                    continue
                else:
                    print(i, fields)
    return source, target, fwd, bwd

# Set the threshold
threshold1 = 0.72  # Adjust as needed
threshold2 = 0.72  # Adjust as needed
cap1 = 1.00
cap2 = 1.00

# Read scores from the comet file
#source, target, scores_fwd, scores_bwd = read_scores('news-commentary/news-commentary-v18.de-en-cometkiwi.tsv')
#source, target, scores_fwd, scores_bwd = read_scores('cc-matrix/cc-matrix-ende-cometkiwi.tsv')
source, target, scores_fwd, scores_bwd = read_scores('europarl/europarl-v10.de-en-cometkiwi.tsv')

filtered_data = [src + '\t' + tgt + '\t' + str(fwd) + '\t' + str(bwd) + '\n' for src, tgt, fwd, bwd in zip(source, target, scores_fwd, scores_bwd) if (fwd >= threshold1) and (fwd <= cap1) and (bwd >= threshold2) and (bwd <= cap2)]

#with open('news-commentary/filtered_' + os.path.basename('news-commentary/news-commentary-v18.de-en-cometkiwi') + '.tsv', 'w') as file:
#with open('cc-matrix/filtered_' + os.path.basename('cc-matrix/cc-matrix-ende-cometkiwi') + '.tsv', 'w') as file:
with open('europarl/filtered_' + os.path.basename('europarl/europarl-v10.de-en-cometkiwi') + '.tsv', 'w') as file:
    file.writelines(filtered_data)

