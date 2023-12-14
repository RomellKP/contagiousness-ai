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