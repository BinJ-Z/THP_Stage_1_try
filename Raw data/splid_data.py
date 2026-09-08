import os, subprocess
import pandas as pd
from sklearn.model_selection import train_test_split

input_file = "/Users/bohe/Desktop/THP CODE2/clean_data/thp_pairs_sequence.csv"
output_folder = "/Users/bohe/Desktop/THP CODE2/split_data"
os.makedirs(output_folder, exist_ok=True)

data = pd.read_csv(input_file).reset_index(drop=True)

input_fasta = os.path.join(output_folder, "temp_peptide.fasta")
output_fasta = os.path.join(output_folder, "temp_cluster.fasta")

with open(input_fasta, "w") as f:
    for i, seq in enumerate(data["sequence"]): f.write(f">seq_{i}\n{seq}\n")

subprocess.run([
    "/Users/bohe/miniconda3/envs/thp/bin/cd-hit",
    "-i", input_fasta,
    "-o", output_fasta,
    "-c", "0.90",
    "-n", "5",
    "-l", "4",
    "-aS", "0.8",
    "-aL", "0.8",
    "-T", "0",
    "-M", "0"
], check=True)

cluster_map = {}
cluster_id = -1

with open(output_fasta + ".clstr") as f:
    for line in f:
        if line.startswith(">Cluster"):
            cluster_id += 1
        else:
            seq_id = int(line.split(">seq_")[1].split("...")[0])
            cluster_map[seq_id] = cluster_id

data["cluster"] = data.index.map(cluster_map)

train_clusters, test_clusters = train_test_split(
    data["cluster"].unique(),
    test_size=0.20,
    random_state=42
)

train = data[data["cluster"].isin(train_clusters)].drop(columns="cluster")
test = data[data["cluster"].isin(test_clusters)].drop(columns="cluster")


train = train.rename(columns={"target_sequence": "Protein_Sequence", "sequence": "Peptide_Sequence"})
test = test.rename(columns={"target_sequence": "Protein_Sequence", "sequence": "Peptide_Sequence"})
train = train[["Protein_Sequence", "Peptide_Sequence", "label"]]
test = test[["Protein_Sequence", "Peptide_Sequence", "label"]]

train.to_csv(os.path.join(output_folder, "train.csv"), index=False)
test.to_csv(os.path.join(output_folder, "test.csv"), index=False)


print("Train:", train.shape)
print("Test:", test.shape)
