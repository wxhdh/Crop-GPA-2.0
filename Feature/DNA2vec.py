from dna2vec.multi_k_model import MultiKModel
import numpy as np
import os
import pandas as pd
from Bio import SeqIO 

filepath = r'dna2vec_4_20.w2v'
mk_model = MultiKModel(filepath)

def generate_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def read_dna_sequences(filepath):
    sequences = []
    for record in SeqIO.parse(filepath, "fasta"):
        sequences.append(str(record.seq)) 
    return sequences

def encode_dna_sequences(sequences, mk_model):
    k_low = mk_model.k_low
    k_high = mk_model.k_high
    all_encoded_vectors = []

    for sequence in sequences:
        encoded_sequence = []
        for k in range(k_low, k_high + 1):
            kmers = generate_kmers(sequence, k)
            for kmer in kmers:
                if kmer in mk_model.data[k].model.vocab:
                    vector = mk_model.vector(kmer)
                    encoded_sequence.extend(vector)
        all_encoded_vectors.append(encoded_sequence)

    return all_encoded_vectors

def process_file(input_file_path, output_file_path, mk_model):
    if not os.path.exists(input_file_path):
        print(f"Input file {input_file_path} does not exist. Skipping...")
        return

    output_directory = os.path.dirname(output_file_path)
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)

    dna_sequences = read_dna_sequences(input_file_path)

    encoded_vectors = encode_dna_sequences(dna_sequences, mk_model)

    df = pd.DataFrame(encoded_vectors)

    df.to_csv(output_file_path, index=False, header=None)
    print(f"Processed file {input_file_path} and saved results to {output_file_path}")

if __name__ == '__main__':
    input_file_path = r'Model/Data/Rice_yield.fasta'
    output_file_path = r'DNA2vec.csv'

    process_file(input_file_path, output_file_path, mk_model)