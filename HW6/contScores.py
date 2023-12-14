import fastaparser
import sys
import pandas as pd 
from datetime import datetime, timedelta


#wasn't sure if you figured out how to get the genome into the csv file
def readFasta(fastaFile):
    headers = []
    genomes = []
    with open(fastaFile, 'r') as fasta:
            parser = fastaparser.Reader(fasta, parse_method = 'quick')
            #adds headers & genomes to arrays
            for seq in parser:
                header = seq.header
                genome = seq.sequence
                headers.append(header)
                genomes.append(genome)
            for i in range(len(headers)):
                headers[i] = headers[i].replace(headers[i][0], "", 1)
    
    return headers, genomes

def convert_to_amino_acids(codon_counts, conversion_table):
    amino_acid_counts = {}
    for codon, count in codon_counts.items():
        amino_acid = conversion_table.get(codon, "Not Found in Conversion Table")
        if amino_acid not in amino_acid_counts:
            amino_acid_counts[amino_acid] = count
        else:
            amino_acid_counts[amino_acid] += count

    return amino_acid_counts

def countAminos(df, genomes):
    codon_to_amino_acid = {}
    with open("codon_table.txt", "r") as file:
        for line in file:
            fields = line.strip().split()
            if len(fields) >= 3:
                codon, amino_acid = fields[:2]
                codon_to_amino_acid[codon] = amino_acid
    cCounts = {}
    for genome in genomes:
        for j in range(0, len(genome[i]), 3):
                codon = genome[i][j : j + 3]
                # Break if codon is too short to be a codon, else, increment counts for this codon
                if len(codon) < 3:
                    break
                elif codon not in cCounts:
                    cCounts[codon] = 1
                else:
                    cCounts[codon] += 1
    aCounts = convert_to_amino_acids(cCounts, codon_to_amino_acid)
    return aCounts

def readCSV(csvFile):
    df = pd.read_csv(csvFile)
    return df

#finds contagiousness score
def findCont(df):
    scoresDict = {}
    df['Release_Date'] = pd.to_datetime(df['Release_Date'])
    for name in df['Accession'].unique():
        name_df = df[df['Accession'] == name]

        startDate = name_df['Release_Date'].min()
        endDate = startDate + timedelta(days=30)

        uploads_count = name_df[(name_df['Release_Date'] >= startDate) & (name_df['Release_Date'] <= endDate)].shape[0]
        scoresDict[name] = uploads_count

    return scoresDict

def main():
    csvFileName = sys.argv[1]
    df = readCSV(csvFileName)
    scoresDict = findCont(df)
    print(scoresDict) 
     
     
if __name__ == "__main__":
     main()