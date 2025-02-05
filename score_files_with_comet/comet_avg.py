# Function to read scores from the comet file
import os.path
import numpy as np
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
            if len(fields) == 1:
                #source.append(fields[0])
                source.append("")
                #target.append(fields[1])
                target.append("")
                fwd.append(float(fields[0]))
                #bwd.append(float(fields[3]))
                bwd.append(0.0)
            else:
                if fields == ['0.4612', '0.4612']:
                    continue
                else:
                    print(i, fields)
    return source, target, fwd, bwd

#folder='en-fr/bt_fromfr/news2019-fromfr/'
#filename='news.2019.fr.shuffled.btfromfr.en-fr.cometkiwi'
folder='en-de/cc-matrix/'
filename='cc-matrix-ende.original.cometkiwi22'
folder='en-de/palm-synthetic/'
filename='greedy_decoded_en_de_blob_level.cometkiwi23'
source, target, scores_fwd, scores_bwd = read_scores(folder+filename+'.sco')

for tresh in [-1000.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84, 0.85]:
    sub_fwd = [fwd for fwd, bwd in zip(scores_fwd, scores_bwd) if fwd >= tresh and fwd <= 1.0] # and fwd < 0.88 and fwd < bwd + 0.03]
    sub_bwd = [bwd for fwd, bwd in zip(scores_fwd, scores_bwd) if bwd >= tresh] # and bwd < 0.88 and fwd > bwd + 0.03]
    print("threshold: ", tresh)
    print("nb: ", len(sub_fwd), "avg score: ", np.mean(sub_fwd))
    if len(sub_bwd) > 0:
        print("nb: ", len(sub_bwd), "avg score: ", np.mean(sub_bwd))

