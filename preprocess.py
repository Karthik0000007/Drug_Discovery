import pandas as pd

# Load raw file
raw_file = 'data\davis.csv'  # your file
data = []

with open(raw_file, 'r') as f:
    for line in f:
        parts = line.strip().split(' ')
        drug_id = parts[0]
        target_id = parts[1]
        smiles = parts[2]
        sequence = parts[3]
        # If the sequence has spaces in it, combine everything except first 3 and last element
        if len(parts) > 5:
            sequence = ' '.join(parts[3:-1])
        affinity = float(parts[-1])
        data.append([drug_id, target_id, smiles, sequence, affinity])

# Convert to DataFrame
df = pd.DataFrame(data, columns=['drug_id', 'target_id', 'smiles', 'sequence', 'affinity'])

# Save to CSV
df.to_csv('data\davis_processed.csv', index=False)
print("Saved processed Davis dataset as davis_processed.csv")

import pandas as pd

# Load raw KIBA file
raw_file = 'data\kiba.csv'  # your file path
data = []

with open(raw_file, 'r') as f:
    for line in f:
        parts = line.strip().split(' ')
        drug_id = parts[0]
        target_id = parts[1]
        smiles = parts[2]
        sequence = parts[3]
        # Combine if sequence has spaces
        if len(parts) > 5:
            sequence = ' '.join(parts[3:-1])
        affinity = float(parts[-1])
        data.append([drug_id, target_id, smiles, sequence, affinity])

# Convert to DataFrame
df = pd.DataFrame(data, columns=['drug_id', 'target_id', 'smiles', 'sequence', 'affinity'])

# Save processed CSV
df.to_csv('data\kiba_processed.csv', index=False)
print("Saved processed KIBA dataset as kiba_processed.csv")


