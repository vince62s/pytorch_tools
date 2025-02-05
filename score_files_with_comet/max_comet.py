import codecs
from argparse import ArgumentParser


parser = ArgumentParser()

parser.add_argument("--topcomet", type=str, help="src repeated n times", required=True)
parser.add_argument(
    "--refcomet", type=str, help="ref repeated n times", required=False
)
parser.add_argument("--output", type=str, help="output file", required=True)

args = parser.parse_args()

with codecs.open(args.refcomet, mode="rb") as file:
    refcomet = [line.decode("utf-8").split('\t') for line in file]

with codecs.open(args.topcomet,mode="rb") as file:
    topcomet = [line.decode("utf-8").split('\t') for line in file]


refcomet_src = [item[0] for item in refcomet]
refcomet_tgt = [item[1] for item in refcomet]
refcomet_score = [float(item[2]) for item in refcomet]

topcomet_src = [item[0] for item in topcomet]
topcomet_tgt = [item[1] for item in topcomet]
topcomet_score = [float(item[2]) for item in topcomet]

output = []
ref_is_max = 0
diff1 = 0
diff2 = 0
with codecs.open(args.output, "w", encoding="utf-8") as output_file:
    for i in range(len(topcomet)):
        if refcomet_src[i] != topcomet_src[i]:
            print("error in src alignment")
            exit()
        if refcomet_score[i] > topcomet_score[i]:
            max_tgt = refcomet_tgt[i]
            max_score = refcomet_score[i]
            ref_is_max += 1
            diff1 += refcomet_score[i] - topcomet_score[i]
            rej_tgt = topcomet_tgt[i]
        else:
            max_tgt = topcomet_tgt[i]
            max_score = topcomet_score[i]
            diff2 += - refcomet_score[i] + topcomet_score[i]
            rej_tgt = refcomet_tgt[i]
        if len(refcomet_src[i]) > 0:
            # output_file.write(refcomet_src[i] + "\t" + max_tgt + "\t" + str(max_score) + "\n")
            output_file.write(max_tgt + " ｟rej_sep｠ " + rej_tgt + "\n")
        
print(ref_is_max / len(topcomet))
print(diff1 / ref_is_max)
print(diff2 / (len(topcomet) - ref_is_max))
