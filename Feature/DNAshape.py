import os
import argparse
import sys
import csv
from deepDNAshape import predictor
for aa in [1]:
    print(aa)
    for x in ["MGW", "Shear", "Stretch", "Stagger", "Buckle", "ProT", "Opening", "Shift", "Slide", "Rise", "Tilt", "Roll", "HelT"]:
        if __name__ == "__main__":
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

            programname = sys.argv[0]
            parser = argparse.ArgumentParser(
                description='Predict DNA shapes for any sequences. Input can be one single sequence or a FILE containing multiple sequences.\n\nFILE format: \nSEQ1\nSEQ2\nSEQ3\n...\n\nExamples:\npython ' + programname + ' --seq AAGGTAGT --feature MGW\
                \npython ' + programname + '.py --file seq.txt --feature Roll --output seq_Roll.csv',
                formatter_class=argparse.RawTextHelpFormatter)
            parser.add_argument("--feature", dest="feature", default=x)
            parser.add_argument("--seq", dest="seq")
            parser.add_argument("--file", dest="file", default=rf"Model/Data/Rice_yield.fasta")
            parser.add_argument("--layer", default=4, dest="layer", type=int)
            parser.add_argument("--output", dest="output", default=rf"{x}.csv")
            parser.add_argument("--batch_size", dest="batch_size", default=2048, type=int)
            parser.add_argument("--gpu", dest="gpu", action="store_true")
            parser.add_argument("--showseq", dest="showseq", action="store_true")
            args = parser.parse_args()
            mode = "cpu" if args.gpu else "gpu"
            myPredictor = predictor.predictor(mode=mode)
            if args.seq:
                prediction = list(map(str, myPredictor.predict(args.feature, args.seq, args.layer)))
                if args.showseq:
                    sys.stdout.write(args.seq + " ")
                sys.stdout.write(" ".join(prediction) + "\n")
            elif args.file:
                outputfa = False
                if args.output == "stdout":
                    usefile = False
                    fout = sys.stdout
                else:
                    usefile = True
                    fout = open(args.output, "w", newline='')
                    writer = csv.writer(fout)

                data = []
                if args.file.endswith(".fa") or args.file.endswith(".fasta"):
                    with open(args.file) as fin:
                        seqname = ""
                        storedSeq = ""
                        for line in fin:
                            if len(line) > 0:
                                if line[0] == ">":
                                    if seqname != "":
                                        data.append((seqname, storedSeq))
                                    seqname = line.strip()
                                    storedSeq = ""
                                else:
                                    storedSeq += line.strip()
                        if storedSeq != "":
                            data.append((seqname, storedSeq))
                else:
                    with open(args.file) as fin:
                        index = 0
                        for seq in fin:
                            data.append((">" + str(index), seq.strip()))

                # Predict batch and output
                seqBatch = []
                seqnames = []
                for seqname, seq in data:
                    seqnames.append(seqname)
                    seqBatch.append(seq.strip())
                    if len(seqBatch) == args.batch_size:
                        prediction = myPredictor.predictBatch(args.feature, seqBatch, args.layer)
                        for i, seq in enumerate(seqBatch):
                            row = [seqnames[i], seq] + list(map(str, prediction[i]))
                            writer.writerow(row)
                        seqBatch = []
                        seqnames = []
                if seqBatch != []:
                    prediction = myPredictor.predictBatch(args.feature, seqBatch, args.layer)
                    for i, seq in enumerate(seqBatch):
                        row = [seqnames[i], seq] + list(map(str, prediction[i]))
                        writer.writerow(row)

                if usefile:
                    fout.close()
            else:
                parser.print_help(sys.stderr)
                sys.exit(1)
