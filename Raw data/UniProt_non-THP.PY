import requests
import pandas as pd
import random
import re

random.seed(42)

# =========================
# 1. 读取正样本
# =========================

thp = pd.read_csv("/Users/bohe/Desktop/THP CODE2/clean_data/thp_clean.csv")

thp["sequence"] = thp["sequence"].astype(str).str.strip().str.upper()
thp["length"] = thp["sequence"].str.len()
thp["label"] = 1

thp_sequences = set(thp["sequence"])


# =========================
# 2. 下载 Swiss-Prot 蛋白
# =========================

url = (
    "https://rest.uniprot.org/uniprotkb/search"
    "?query=reviewed:true"
    "&format=tsv"
    "&fields=sequence"
    "&size=500"
)

proteins = []

while url and len(proteins) < 3000:

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    lines = r.text.strip().split("\n")[1:]

    for line in lines:
        if line:
            proteins.append(line)

    link = r.headers.get("Link", "")
    match = re.search(r'<(.+)>; rel="next"', link)
    url = match.group(1) if match else None


# =========================
# 3. 生成同长度负样本
# =========================

standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
negative = thp.copy()
negative_sequences = set()

for i, row in negative.iterrows():

    length = row["length"]

    while True:

        protein = random.choice(proteins).upper()

        if len(protein) < length:
            continue

        start = random.randint(0, len(protein) - length)
        peptide = protein[start:start + length]

        if not set(peptide).issubset(standard_aa):
            continue

        if peptide in thp_sequences:
            continue

        if peptide in negative_sequences:
            continue

        negative.at[i, "sequence"] = peptide
        negative_sequences.add(peptide)
        break


negative["label"] = 0


# =========================
# 4. 合并正负样本
# =========================

all_pairs = pd.concat([thp, negative], ignore_index=True)


# =========================
# 5. 保存
# =========================

all_pairs.to_csv(
    "/Users/bohe/Desktop/THP CODE2/clean_data/thp_pairs.csv",
    index=False
)
