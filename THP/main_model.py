import torch
import torch.nn as nn
import esm
import os
import time
import pandas as pd
from functools import reduce
from collections import defaultdict
from Bio import SeqIO
import iFeatureOmegaCLI
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from Bio.Align import substitution_matrices
import sys
import subprocess
import metapredict as meta
import tempfile
import pickle
from sklearn.preprocessing import StandardScaler
device = torch.device("cpu")

def create_protein_peptide_dataset(train_file, test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    return train, test

def global_features(df, descriptors=None):

    if descriptors is None:
        descriptors = ["AAC", "PAAC", "APAAC", "CTDC", "CTDT", "CTDD"]

    data = df.copy().reset_index(drop=True)
    original_columns = data.columns.tolist()

    for sequence_type in ["Protein", "Peptide"]:

        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            for i, sequence in enumerate(data[f"{sequence_type}_Sequence"]):
                f.write(f">{i}\n{sequence}\n")
            temp_fasta = f.name

        try:
            for descriptor in descriptors:

                pro = iFeatureOmegaCLI.iProtein(temp_fasta)
                pro.get_descriptor(descriptor)

                if pro.encodings is None:
                    raise ValueError(
                        f"{sequence_type} {descriptor} failed: {pro.error_msg}"
                    )

                feature = pro.encodings.copy().reset_index(drop=True)

                feature.columns = [
                    f"{sequence_type}_{descriptor}_{col}"
                    for col in feature.columns
                ]

                data = pd.concat([data, feature], axis=1)

        finally:
            os.remove(temp_fasta)

    feature_columns = [col for col in data.columns if col not in original_columns]

    return data, feature_columns


def standardize_global_features(train, test, feature_columns):
    scaler = StandardScaler()
    train[feature_columns] = scaler.fit_transform( train[feature_columns])
    test[feature_columns] = scaler.transform( test[feature_columns])
    return train, test, scaler


#  ESM2 encoder，参数全部冻结，是对的吧，我们样本量不大，部分冻结好吗？



class ESM(nn.Module):
    def __init__(self):
        super().__init__()
        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        for param in self.esm_model.parameters(): param.requires_grad = False
        self.our_converter = self.alphabet.get_batch_converter()

    def encode(self, seqs):
        data = [(f"seq{i}", seq) for i, seq in enumerate(seqs)]
        _, _, batch_tokens = self.our_converter(data)
        batch_tokens = batch_tokens.to(next(self.esm_model.parameters()).device)
        self.esm_model.eval()

        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)

        embedding = results["representations"][33][:, 1:]
        tokens = batch_tokens[:, 1:]

        mask = (tokens != self.alphabet.padding_idx) & (tokens != self.alphabet.eos_idx)
        embedding = embedding * mask.unsqueeze(-1).to(embedding.dtype)
        return embedding, mask

    def forward(self, prot_seqs, pep_seqs):
        prot_embedding, prot_mask = self.encode(prot_seqs)
        pep_embedding, pep_mask = self.encode(pep_seqs)

        return prot_embedding, pep_embedding, prot_mask, pep_mask


class CNNPool(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim1=512, hidden_dim2=256, output_dim=256, window_size=5):
        super().__init__()
        pad = (window_size - 1) // 2
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim1, window_size, padding=pad)
        self.conv2 = nn.Conv1d(hidden_dim1, hidden_dim2, window_size, padding=pad)
        self.conv3 = nn.Conv1d(hidden_dim2, output_dim, window_size, padding=pad)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, embedding, mask):
        output = embedding.permute(0, 2, 1)
        mask = mask.unsqueeze(1)  # mask目前是单独的一张对应的表

        output = self.relu(self.conv1(output))
        output = output * mask

        output = self.relu(self.conv2(output))
        output = output * mask

        output = self.relu(self.conv3(output))
        output = output * mask

        return self.pool(output).squeeze(-1)




# AAIndex + BLOSUM62 + ZScale + Disorder


def get_residue_features(df, sequence_type, save_file=None):

    all_features = {}

    for i, sequence in enumerate(df[f"{sequence_type}_Sequence"]):

        sequence = str(sequence).upper()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(f">{i}\n{sequence}\n")
            temp_fasta = f.name

        try:
            pro = iFeatureOmegaCLI.iProtein(temp_fasta)
            residue_features = []

            for descriptor in ["AAIndex", "BLOSUM62", "ZScale"]:
                pro.get_descriptor(descriptor)

                if pro.encodings is None:
                    raise ValueError( f"{sequence_type} {i} {descriptor} failed: {pro.error_msg}")
                residue_features.append( pro.encodings.to_numpy(dtype=np.float32).reshape(len(sequence), -1))

            disorder = np.asarray(meta.predict_disorder(sequence), dtype=np.float32 ).reshape(-1, 1)
            residue_features.append(disorder)
            all_features[i] = np.concatenate(residue_features, axis=1)

        finally:
            os.remove(temp_fasta)

    if save_file is not None:
        with open(save_file, "wb") as f:
            pickle.dump(all_features, f)

    return all_features



def fit_residue_scaler(protein_features, peptide_features):
    all_residues = np.concatenate(list(protein_features.values()) + list(peptide_features.values()), axis=0)
    mean = all_residues.mean(axis=0)
    std = all_residues.std(axis=0)
    std[std == 0] = 1
    return mean, std


def standardize_residue_features(features, mean, std):
    return {seq_id: ((x - mean) / std).astype(np.float32) for seq_id, x in features.items()}



def build_interaction_matrix(protein_features, peptide_features):
    protein_features = torch.as_tensor(protein_features, dtype=torch.float32)
    peptide_features = torch.as_tensor(peptide_features, dtype=torch.float32)

    product = protein_features[:, None, :] * peptide_features[None, :, :]
    difference = torch.abs(protein_features[:, None, :] - peptide_features[None, :, :])

    return torch.cat([product, difference], dim=2).permute(2, 0, 1).unsqueeze(0)


class Interaction2DCNN(nn.Module):
    def __init__(self, input_channels, output_dim=256):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(input_channels, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, output_dim, 3, padding=1)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, interaction):
        output = self.relu(self.conv1(interaction))
        output = self.relu(self.conv2(output))
        output = self.relu(self.conv3(output))

        return self.pool(output).flatten(1)


# Global physicochemical feature branch


class GlobalFeatureEncoder(nn.Module):

    def __init__(self, input_dim, output_dim=256):
        super(GlobalFeatureEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, global_features):
        return self.encoder(global_features)


#  Main Model


class MainModel(nn.Module):
    def __init__(self, residue_dim, global_dim):
        super().__init__()

        self.ESM = ESM()
        self.prot_cnn = CNNPool(input_dim=1280, output_dim=256)
        self.pep_cnn = CNNPool(input_dim=1280, output_dim=256)

        self.interaction_cnn = Interaction2DCNN(input_channels=residue_dim * 2, output_dim=256)
        self.global_encoder = GlobalFeatureEncoder(input_dim=global_dim, output_dim=256)

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, protein_sequences, peptide_sequences, protein_residue_features, peptide_residue_features, global_features):

        prot_embedding, pep_embedding, prot_mask, pep_mask = self.ESM(protein_sequences, peptide_sequences)

        prot_vector = self.prot_cnn(prot_embedding, prot_mask)
        pep_vector = self.pep_cnn(pep_embedding, pep_mask)

        interaction_vectors = []

        for prot_res, pep_res in zip(protein_residue_features, peptide_residue_features):
            interaction = build_interaction_matrix(prot_res, pep_res)
            interaction = interaction.to(next(self.interaction_cnn.parameters()).device)
            interaction_vectors.append(self.interaction_cnn(interaction))

        interaction_vector = torch.cat(interaction_vectors, dim=0)
        global_vector = self.global_encoder(global_features)

        combined = torch.cat([prot_vector, pep_vector, interaction_vector, global_vector], dim=1)
        logits = self.classifier(combined)

        return logits.squeeze(1)

    
