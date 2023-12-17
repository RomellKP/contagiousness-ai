import fastaparser
import sys
import pandas as pd 
from datetime import datetime, timedelta


### Reads a fasta file as headers (>), genomes
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

### Reads a file to get the amino acids conversion table
def get_amino_table(filename):
    conversion_table = {}
    with open(filename, "r") as file:
        for line in file:
            fields = line.strip().split()
            if len(fields) >= 3:
                codon, amino_acid, letter = fields[:3]
                conversion_table[codon] = letter
    return conversion_table

### Converts codon counts into amino acids counts using the conversion table
def convert_to_amino_acids(codon_counts, conversion_table):
    amino_acid_counts = {'A': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0, 'G': 0, 'H': 0, 'I': 0, 'K': 0,
                   'L': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0, "Other" : 0}
    for codon, count in codon_counts.items():
        amino_acid = conversion_table.get(codon, "Other")
        if amino_acid not in amino_acid_counts:
            continue  # can alter later if necessary
        else:
            amino_acid_counts[amino_acid] += count

    return amino_acid_counts

# Returns a dictionary like : aminoCounts = {header1: {leucine: 10, proline: 10}, header2: {leusine: 10, proline: 10}}
def countAminos(headers, genomes, conversionTable):
    aminoCounts = {}
    for i in range(len(genomes)):
        if headers[i] not in aminoCounts:
            cCounts = {}
            for j in range(0, len(genomes[i]), 3):
                    codon = genomes[i][j : j + 3]
                    # Break if codon is too short to be a codon, else, increment counts for this codon
                    if len(codon) < 3:
                        break
                    elif codon not in cCounts:
                        cCounts[codon] = 1
                    else:
                        cCounts[codon] += 1
            aCounts = convert_to_amino_acids(cCounts, conversionTable)
            # print(aCounts)
            aminoCounts[headers[i]] = aCounts
    return aminoCounts

def readCSV(csvFile):
    df = pd.read_csv(csvFile)
    return df

# Finds contagiousness score
def findCont(df):
    scoresDict = {}
    df['Release_Date'] = pd.to_datetime(df['Release_Date'])
    for name in df['Accession']:
        name_df = df[df['Accession'] == name]

        startDate = name_df['Release_Date'].min()
        endDate = startDate + timedelta(days=30)

        uploads_count = name_df[(name_df['Release_Date'] >= startDate) & (name_df['Release_Date'] <= endDate)].shape[0]
        scoresDict[name] = uploads_count

    return scoresDict

### Creates a data frame from data organized like: headers = [Accession,  Release_Date, Aminos(20)... Other, Contagiousness_Score
def createDF(countsDict):
    data = []
    for metadata, values in countsDict.items():
        meta = metadata.split('|')
        accession_number = meta[0].strip()
        release_date = meta[1].strip()

        row = {'Accession': accession_number, 'Release_Date': release_date, **values}
        data.append(row)
    # Write as a dataframe to csv
    df = pd.DataFrame(data)
    # Filter duplicates by accession number
    df = df.drop_duplicates(subset=['Accession'])
    # Get contagiousness scores, add to dictionary
    contagiousness_scores = findCont(df)
    df['Contagiousness_Score'] = df['Accession'].map(contagiousness_scores).fillna(0).astype(int)
    return df



def main():
    """
    Reads in a fasta file, cleans it, and writes as readable dataframe to csv
    Usage: python3 dfCreator.py
    Example: python3 dfCreator.py ../data/raw_data.fasta ../data/processed_data.csv
    """

    inputFileName = sys.argv[1]
    outputFileName = sys.argv[2]

    # Read user's input file, get conversion table
    headers, genomes = readFasta(inputFileName)
    convTable = get_amino_table("../data/codon_table.txt")

    # Count amino acids in data using conversion table
    countsDict = countAminos(headers, genomes, convTable)
    
    # Make dataframe, write to csv
    df = createDF(countsDict)
    df.to_csv(outputFileName, index=False)
     
     
if __name__ == "__main__":
     main()
