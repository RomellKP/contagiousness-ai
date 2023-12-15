import pandas as pd
from matplotlib import pyplot as plt
import csv
import sys
import numpy as np
### ---------------------------------------------------------------------------------------------------------------- ###

# Returns a dictionary of codons:counts from an array of codons and counts
def get_codon_dict(x, y):
    codon_counts = {}
    for i in range(len(x)):
        if x[i] not in codon_counts:
            codon_counts[x[i]] = y[i]
        else:
            codon_counts[x[i]] += y[i]
    return codon_counts

# Reads codons & their counts in from a csv and returns arrays [codon], [count]
def readCodonsIn(filename):
    x, y = [], []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row !=[]:
                codon = row[0]
                count = row[1]
                x.append(codon)
                y.append(int(count))
    return x, y

# Converts dictionary of codons:counts into dictionary of amino acids:counts
def convert_to_amino_acids(codon_counts, conversion_table):
    amino_acid_counts = {}
    for codon, count in codon_counts.items():
        amino_acid = conversion_table.get(codon, "Not Found in Conversion Table")
        if amino_acid not in amino_acid_counts:
            amino_acid_counts[amino_acid] = count
        else:
            amino_acid_counts[amino_acid] += count

    return amino_acid_counts

# Convert a dictionary to a numpy array
def dict_to_npArr(myDict):
    x, y = [], []
    # Populate as lists (faster, I think):
    for key, val in myDict.items():
        x.append(key)
        y.append(val)
    # Convert into numpy arrays
    x = np.array(x)
    y = np.array(y)
    return x, y

# Pairplot where x1 & y1 are the leading plot pairs
def pairplot(x1, y1, x2, y2):
    # Set window size, bar width
    x = np.arange(len(x1))
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the width (10) and height (6) as needed

    width = 0.4  
    # Declare bars for plot
    bars1 = ax.bar(x - width/2, y1, width, label='Separate Genome File (correct frame shift)')
    bars2 = ax.bar(x + width/2, y2, width, label='Whole Genome File (random frame shift)')

    # Set ticks to be vertical, add labels, plot
    plt.xticks(x, x1, rotation='vertical')
    ax.set_xlabel("Amino Acids")
    ax.set_ylabel("Frequencies")
    ax.set_title("Amino Acid Frequencies")
    ax.legend()

    plt.show()

# Sort arrays according to x1 & x2's order
def sortArrays(x1, y1, x2, y2):
    sorted_indices = np.argsort(y1)
    sorted_indices = sorted_indices[::-1]
    print(x1)
    print(y1)
    x1 = x1[sorted_indices]
    y1 = y1[sorted_indices]
    x1 = list(x1)
    y1 = list(y1)
    x2 = list(x2)
    y2 = list(y2)
    ## x1 now contains the codons from separate file, now sorted by frequency

    # Sort the control array to match new_x:
    new_x2 = []
    new_y2 = []
    for i in range(len(x1)):
        try:
            ind = x2.index(x1[i])
            new_x2.append(x2[ind])
            new_y2.append(y2[ind])
        except:
            continue
    # Add any elements that are in the control (separate genome file) but not the new array (full genome random window file)
    for i in range(len(x2)):
        if x2[i] not in new_x2:
            new_x2.append(x2[i])
            new_y2.append(y2[i])
    return x1, y1, new_x2, new_y2
### ---------------------------------------------------------------------------------------------------------------- ###

### Creates a side-by-side bar plot of codons & their frequencies
def plot_codons(correct_window_filename, random_window_filename, amino_acids=True):
    # Read codons from CSVs, count into dictionaries
    correct_x, correct_y = readCodonsIn(correct_window_filename)
    random_x, random_y = readCodonsIn(random_window_filename)
    correct_window_codon_counts = get_codon_dict(correct_x, correct_y)
    random_window_codon_counts = get_codon_dict(random_x, random_y)
    
    # Read codon to amino acid conversion table, convert into dictionary of counts
    codon_to_amino_acid = {}
    with open("codon_table.txt", "r") as file:
        for line in file:
            fields = line.strip().split()
            if len(fields) >= 3:
                codon, amino_acid = fields[:2]
                codon_to_amino_acid[codon] = amino_acid


    correct_amino_acid_counts = convert_to_amino_acids(correct_window_codon_counts, codon_to_amino_acid)
    random_amino_acid_counts = convert_to_amino_acids(random_window_codon_counts, codon_to_amino_acid)

    print(correct_amino_acid_counts)
    print(random_amino_acid_counts)
    ## Convert dictionaries to numpy arrays for sorting:
    correct_x, correct_y = dict_to_npArr(correct_amino_acid_counts)
    random_x, random_y = dict_to_npArr(random_amino_acid_counts)

    ## Sort arrays according to x1 & x2's order
    correct_x, correct_y, random_x, random_y = sortArrays(correct_x, correct_y, random_x, random_y)
    pairplot(correct_x, correct_y, random_x, random_y)




# Check that user gave a file name
if len(sys.argv) != 3:
    print("Usage: python3 plot_codons.py <control_csv_filename> <new_csv_filename>")
    sys.exit(1)

# Get the filenames from the command-line arguments
control_filename = sys.argv[1]
new_filename = sys.argv[2]
plot_codons(control_filename, new_filename)

