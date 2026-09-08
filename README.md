<img width="9070" height="5290" alt="预测模型1_01" src="https://github.com/user-attachments/assets/dffbb98b-6453-49e4-a99e-4ae984e664c5" />Module
1.ESM2
Protein and peptide sequences are encoded by frozen ESM2-650M→1D CNN  
2. Protein–peptide interaction matrix
AAIndex, BLOSUM62, ZScale, and Disorder are extracted for each residue→2D CNN 
3.Global Sequence Features
AAC, PAAC, APAAC, CTDC, CTDT, and CTDD are calculated  →MLP 
4.Feature Fusion and Prediction
Concatenate all vectors → 1024-D.
Fully connected layers: 1024 → 512 → 128 → 1
Sigmoid → interaction probability.

Problem:
1: Now use all amino acid sequence
Change it to specifying which amino acid regions of each target protein actually interact with the peptide requires checking the information in each literature.

2: The THP database does not indicate whether the target proteins are homo or mice. Therefore, searching for amino acid sequences for target protein name on Swiss cannot determine whether it is homo or mice. 

3:  I  frozen all the parameters in protein language ESM2 model? Our data only includes 400 positive samples and 400 negative samples.


