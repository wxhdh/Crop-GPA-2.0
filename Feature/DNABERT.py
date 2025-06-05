import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import csv
import os

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

fasta_file = 'Model/Data/Rice_yield.fasta'
sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]

embedding_mean_path = "DNABERT.csv"

if not os.path.isfile(embedding_mean_path):
    with open(embedding_mean_path, "a", newline="") as mean_csv_file:
        mean_csv_writer = csv.writer(mean_csv_file)

for i, dna in enumerate(sequences, start=1):
    print(dna)
    print(f"Processing sequence {i}/{len(sequences)}")
    inputs = tokenizer(dna, return_tensors='pt').to(device)
    with torch.no_grad():
        hidden_states = model(**inputs)[0]
    embedding_mean = torch.mean(hidden_states[0], dim=0).cpu().numpy()

    with open(embedding_mean_path, "a", newline="") as mean_csv_file:
        mean_csv_writer = csv.writer(mean_csv_file)
        mean_csv_writer.writerow(embedding_mean.tolist())
