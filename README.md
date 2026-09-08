<img width="9070" height="5290" alt="预测模型1_01" src="https://github.com/user-attachments/assets/dffbb98b-6453-49e4-a99e-4ae984e664c5" />

## Module

### 1. ESM2

Protein and peptide sequences are encoded by frozen ESM2-650M → 1D CNN.

### 2. Protein–Peptide Interaction Matrix

AAIndex, BLOSUM62, ZScale, and Disorder are extracted for each residue → 2D CNN.

### 3. Global Sequence Features

AAC, PAAC, APAAC, CTDC, CTDT, and CTDD are calculated → MLP.

### 4. Feature Fusion and Prediction

Concatenate all vectors → 1024-D.
Fully connected layers: 1024 → 512 → 128 → 1.
Sigmoid → interaction probability.

## Problems

### 1. Currently using the full amino acid sequence

The model currently uses the full amino acid sequence of each target protein.

To change this to specific amino acid regions of each target protein that actually interact with the peptide, the interaction region needs to be identified by checking the information reported in the corresponding literature.

### 2. Species information is missing in the THP database

The THP database does not indicate whether the target proteins are from humans or mice.

Therefore, searching Swiss-Prot using only the target protein name cannot determine whether the retrieved amino acid sequence belongs to human or mouse.

### 3. I  frozen all the parameters in protein language ESM2 model? Our data only includes 400 positive samples and 400 negative samples.

