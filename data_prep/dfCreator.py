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

# Returns a dictionary like : aminoCounts = {header1: {leucine: 10, proline: 10}, header2: {leusine: 10, proline: 10}}
def countAminos(headers, genomes):
    codon_to_amino_acid = {}
    with open("codon_table.txt", "r") as file:
        for line in file:
            fields = line.strip().split()
            if len(fields) >= 3:
                codon, amino_acid = fields[:2]
                codon_to_amino_acid[codon] = amino_acid
    aminoCounts = {}
    for i in range(len(genomes)):
        if headers[i] not in aminoCounts:
            cCounts = {}
            for j in range(0, len(genomes[i]), 3):
                    codon = genomes[j : j + 3]
                    # Break if codon is too short to be a codon, else, increment counts for this codon
                    if len(codon) < 3:
                        break
                    elif codon not in cCounts:
                        cCounts[codon] = 1
                    else:
                        cCounts[codon] += 1
            aCounts = convert_to_amino_acids(cCounts, codon_to_amino_acid)
            aminoCounts[headers[i]] = aCounts
    return aminoCounts

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

#creates a data frame from data organized like: data = [{Label: ***, Contagiousness_Score: ***, proline: ***, ...}, {Label: ***, ...}, ...]
def createDF(headers, genomes, scoresDict):
    data = []
    aminoCounts = countAminos(headers, genomes)
    for i in range(len(headers)):
        if headers[i] in scoresDict:
            curAminoCount = aminoCounts[headers[i]]
            curAminoCount['Label'] = headers[i]
            curAminoCount['Contagiousness_Score'] = scoresDict[headers[i]]
            data.append(curAminoCount)
    newDF = pd.DataFrame(data)
    return newDF


def main():
    csvFileName = sys.argv[1]
    fastaFileName = sys.argv[2]
    df = readCSV(csvFileName)
    scoresDict = findCont(df)
    print(scoresDict)
    headers, genomes = readFasta(fastaFileName)
    newDF = createDF(df, headers, genomes, scoresDict)
    print(newDF) 
     
     
if __name__ == "__main__":
     main()