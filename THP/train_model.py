#conda activate thp
#cd "/Users/bohe/Desktop/THP CODE2"
#python train_model.py

from main_model import *
train, test = create_protein_peptide_dataset("/Users/bohe/Desktop/THP CODE2/split_data/train1.csv", "/Users/bohe/Desktop/THP CODE2/split_data/test1.csv")

train, feature_columns = global_features(train)
test, _ = global_features(test)
train, test, global_scaler = standardize_global_features(train, test, feature_columns)

train_protein_residue = get_residue_features(train, "Protein")
train_peptide_residue = get_residue_features(train, "Peptide")
test_protein_residue = get_residue_features(test, "Protein")
test_peptide_residue = get_residue_features(test, "Peptide")

residue_mean, residue_std = fit_residue_scaler(train_protein_residue, train_peptide_residue)
train_protein_residue = standardize_residue_features(train_protein_residue, residue_mean, residue_std)
train_peptide_residue = standardize_residue_features(train_peptide_residue, residue_mean, residue_std)
test_protein_residue = standardize_residue_features(test_protein_residue, residue_mean, residue_std)
test_peptide_residue = standardize_residue_features(test_peptide_residue, residue_mean, residue_std)

residue_dim = next(iter(train_protein_residue.values())).shape[1]
global_dim = len(feature_columns)

model = MainModel(residue_dim, global_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

epochs, batch_size = 20, 4

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for start in range(0, len(train), batch_size):
        batch = train.iloc[start:start + batch_size]
        protein_sequences = batch["Protein_Sequence"].tolist()
        peptide_sequences = batch["Peptide_Sequence"].tolist()
        protein_residue_features = [train_protein_residue[i] for i in batch.index]
        peptide_residue_features = [train_peptide_residue[i] for i in batch.index]
        global_feature_tensor = torch.tensor(batch[feature_columns].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
        labels = torch.tensor(batch["label"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)

        optimizer.zero_grad()
        logits = model(protein_sequences, peptide_sequences, protein_residue_features, peptide_residue_features, global_feature_tensor)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)

    print(f"Epoch {epoch + 1}/{epochs} Loss: {total_loss / len(train):.4f}")

model.eval()
all_probs = []

with torch.no_grad():
    for start in range(0, len(test), batch_size):
        batch = test.iloc[start:start + batch_size]
        protein_sequences = batch["Protein_Sequence"].tolist()
        peptide_sequences = batch["Peptide_Sequence"].tolist()
        protein_residue_features = [test_protein_residue[i] for i in batch.index]
        peptide_residue_features = [test_peptide_residue[i] for i in batch.index]
        global_feature_tensor = torch.tensor(batch[feature_columns].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)

        logits = model(protein_sequences, peptide_sequences, protein_residue_features, peptide_residue_features, global_feature_tensor)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())

test["probability"] = all_probs
test["prediction"] = (test["probability"] >= 0.5).astype(int)
test.to_csv("/Users/bohe/Desktop/THP CODE2/test_prediction.csv", index=False)
