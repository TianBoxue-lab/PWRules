from Bio import SeqIO


def read_fasta(path):
    uniprot2seq = {}
    for record in SeqIO.parse(path, "fasta"):
        uniprot_id = record.id
        if '|' in uniprot_id:
            uniprot_id = uniprot_id.split('|')[1]
        uniprot2seq[uniprot_id] = str(record.seq)
    return uniprot2seq

