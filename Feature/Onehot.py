import csv
from Bio import SeqIO

def convert_to_POEN(sequence):
    """
    Convert a DNA sequence into a POEN (one-hot) encoding.
    """
    z = []
    for y in sequence:
        if y == "A" or y == "a":
            z.append([1, 0, 0, 0])
        elif y == "T" or y == "t":
            z.append([0, 1, 0, 0])
        elif y == "G" or y == "g":
            z.append([0, 0, 0, 1])
        elif y == "C" or y == "c":
            z.append([0, 0, 1, 0])
    return z

def write_vectors_to_csv(vectors, filename, row_size):
    """
    Write POEN encoded vectors to a CSV file with specified row size.
    """
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Initialize an empty list for constructing rows
        full_row = []
        for vector in vectors:
            full_row.extend(vector)
            # Write rows to CSV when they reach the specified size
            while len(full_row) >= row_size:
                csvwriter.writerow(full_row[:row_size])
                full_row = full_row[row_size:]
        # Write any remaining elements to the final row
        if full_row:
            csvwriter.writerow(full_row)

def process_fasta_file(input_file_path, output_file_path, row_size):
    """
    Read sequences from a FASTA file, convert them to POEN encoding, and write to a CSV file.
    """
    # Read sequences from FASTA file
    sequences = [str(record.seq) for record in SeqIO.parse(input_file_path, "fasta")]

    # Convert sequences to POEN encoding
    encoded_vectors = []
    for seq in sequences:
        encoded_vectors.extend(convert_to_POEN(seq))

    # Write the encoded vectors to a CSV file
    write_vectors_to_csv(encoded_vectors, output_file_path, row_size)

if __name__ == '__main__':
    # Input and output file paths
    input_file_path = 'Model/Data/Rice_yield.fasta'
    output_file_path = 'Onehot.csv'
    row_size = 164  # Set the row size for CSV output

    # Process the FASTA file and write the results to CSV
    process_fasta_file(input_file_path, output_file_path, row_size)
