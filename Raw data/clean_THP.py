import os
import unicodedata
import pandas as pd

input_file = "/Users/bohe/Desktop/THP CODE2/raw data/main.csv"
output_folder = "/Users/bohe/Desktop/THP CODE2/clean_data"
os.makedirs(output_folder, exist_ok=True)

thp_raw = pd.read_csv(input_file)
print("Original data shape:", thp_raw.shape)


standard_aa = set("ACDEFGHIKLMNPQRSTVWY")

thp_clean = thp_raw.dropna(subset=["sequence"]).copy()
thp_clean["sequence"] = thp_clean["sequence"].astype(str).str.strip().str.upper()


thp_clean = thp_clean[
    thp_clean["sequence"].apply(
        lambda seq: len(seq) > 0 and set(seq).issubset(standard_aa)
    )
].copy()

# 长度
thp_clean["length"] = thp_clean["sequence"].str.len()

# 只保留 5-20 aa
thp_clean = thp_clean[
    thp_clean["length"].between(5, 20)
].copy()

# 注意：这里不要按 sequence 去重
# 因为一个 THP 可以对应多个 target


thp_clean["receptors_biomarker_clean"] = thp_clean["receptors_biomarker"].apply(
    lambda x: unicodedata.normalize("NFKC", str(x)).strip()
    if pd.notna(x) else pd.NA
)

thp_clean["receptors_biomarker_clean"] = (
    thp_clean["receptors_biomarker_clean"]
    .str.replace(r"\s+", " ", regex=True)
    .str.replace(r"\s+,", ",", regex=True)
    .str.replace(r",\s*", ", ", regex=True)
    .replace(
        [
            "",
            "NA",
            "N/A",
            "None",
            "none",
            "Unknown",
            "unknown",
            "-",
            "null"
        ],
        pd.NA
    )
)

# 删除没有 target 的记录
thp_clean = thp_clean.dropna(  subset=["receptors_biomarker_clean"]).copy()


receptor_mapping = {

    # EGFR / HER2
    "Epidermal growth factor receptor": "EGFR",
    "EFGR": "EGFR",

    "HER2": "ERBB2",
    "HER2 receptor": "ERBB2",
    "Her-2": "ERBB2",
    "ErbB2": "ERBB2",
    "ErbB-2": "ERBB2",
    "Her-2 and ErbB-2": "ERBB2",
    "HER2(Human epidermal growth factor receptor 2)": "ERBB2",

    # Neuropilin
    "NRP-1": "NRP1",
    "Neuropilin-1": "NRP1",
    "neuropilin-1": "NRP1",
    "neuropilin- 1": "NRP1",
    "Neuropilin-1 (NRP1)": "NRP1",
    "Neuropilin-1 (NRP-1)": "NRP1",
    "neuropilin-1 (NRP1)": "NRP1",
    "neuropilin-1 (NRP-1)": "NRP1",
    "neuropilin 1 (NRP-1)": "NRP1",
    "NRP-1 b1b2": "NRP1",
    "Wild Type NRP-1": "NRP1",

    "Neuropilin-2": "NRP2",
    "neuropilin-2": "NRP2",
    "NRP-2": "NRP2",

    # VEGFR
    "VEGFR2": "KDR",
    "VEGFR-2": "KDR",
    "vascular endothelial growth factor receptor 2 (VEGFR2)": "KDR",

    "VEGFR3": "FLT4",
    "VEGFR-3": "FLT4",
    "VGFR3": "FLT4",
    "VEGFR-3 (Human vascular endothelia growth factor receptor 3 )": "FLT4",
    "VEGFR-3 (Human vascular endothelia growth factor receptor 3)": "FLT4",

    # CD13 / APN
    "CD13": "ANPEP",
    "APN": "ANPEP",
    "Aminopeptidase N": "ANPEP",
    "aminopeptidase N": "ANPEP",
    "Aminopeptidase N (CD13)": "ANPEP",
    "CD13 (aminopeptidase N)": "ANPEP",
    "CD13/ANPEP": "ANPEP",
    "APN, a membrane-bound aminopeptidase": "ANPEP",

    # p32
    "p32": "C1QBP",
    "P32": "C1QBP",
    "p32 protein": "C1QBP",
    "p32 receptor": "C1QBP",
    "p32 mitochondrial protein": "C1QBP",
    "Cell surface p32": "C1QBP",
    "gC1q/p32 receptor": "C1QBP",
    "p32/gC1qR": "C1QBP",
    "p32 and gC1qR": "C1QBP",
    "p32 and gC1gR": "C1QBP",
    "p32, p33, gC1qR, HABP1 (p32)": "C1QBP",

    # GRP78
    "GRP78": "HSPA5",
    "grp78": "HSPA5",
    "Glucose-regulated protein 78 (GRP78)": "HSPA5",

    # PSMA
    "PSMA": "FOLH1",
    "Prostate-specific membrane antigen": "FOLH1",
    "Prostate-specific membrane antigen (PSMA)": "FOLH1",
    "PSMA, Prostrate specific membrane antigen": "FOLH1",

    # PD-L1
    "PD-L1": "CD274",
    "PDL1": "CD274",

    # Other proteins
    "uPAR": "PLAUR",
    "urokinase-type plasminogen activator receptor (uPAR)": "PLAUR",

    "Nucleolin": "NCL",
    "Nucleoin": "NCL",

    "E-selectin": "SELE",

    "Podoplanin": "PDPN",

    "Vitronectin": "VTN",

    "Galectin-3": "LGALS3",

    "MT1-MMP(Membrane type 1Matrix metalloproteinase )": "MMP14",

    "MMP-11 proteinase": "MMP11",
    "MMP2 protein": "MMP2",

    "MECA32 (Second endothelial marker, the cell-surface antigen)": "PLVAP",

    "Membrane-bound proline-specific aminopeptidase P": "XPNPEP2",
    "Aminopeptidase P (XPNPEP2)": "XPNPEP2",

    "membrane dipeptidase (MDP)": "DPEP1",

    "Secreted clusterin": "CLU",

    "fibronectin": "FN1",

    "vimentin": "VIM",
    "Vimentin": "VIM",
    "tumor-specific ectopic expression of vimentin": "VIM",

    "annexin 1": "ANXA1",
    "Anxa1": "ANXA1",

    "folate receptor alpha (FRα)": "FOLR1",

    "CD206/MRC1": "MRC1",
    "CD206 (Mannose receptor)": "MRC1",

    "Glypican-3 (GPC3)": "GPC3",

    "ICAM-1 receptors": "ICAM1",

    "Transferrin": "TF",
    "Transferrin Receptors": "TFRC",
    "Transferrin receptor (TfR)": "TFRC",

    "CD-22": "CD22",

    "CD31": "PECAM1",
    "PECAM-1 (platelet endothelial cell adhesion molecule-1)": "PECAM1",

    "VEGF": "VEGFA",
    "VEGF-C": "VEGFC",

    "APA": "ENPEP",

    "LHRH(Receptors for luteinizing hormone-releasing hormone)": "GNRHR",

    "Gastrin-releasing peptide receptor": "GRPR",
    "Gastrin releasing peptide receptor": "GRPR",
    "gastrin releasing peptide receptor (GRPr)": "GRPR",
    "released peptide (GRP) receptor": "GRPR",

    "cholecystokinin B(CCKB) receptor": "CCKBR",

    "Constitutive androstane receptor (CAR)": "NR1I3",

    "Mesothelin": "MSLN",

    "prohibitin": "PHB",

    "mono carboxylate transporter 1 (MCT1)": "SLC16A1",

    "Frizzled (FZD-5) receptors": "FZD5",

    "Cadherin 2": "CDH2",

    "Four-and-a-half LIM-only protein 3 (FHL3)": "FHL3",
    "Four-and-a-half LIM-only protein 3, FHL3": "FHL3",

    "Interleukin-4 Receptor(IL4R)": "IL4R",
    "IL-4": "IL4",

    "IL-13Ra2": "IL13RA2",
    "Interleukin 13 receptor 2 (IL-13R 2)": "IL13RA2",

    "KIM-1(Kidney injury molecule-1)": "HAVCR1",

    "EphB4": "EPHB4",
    "EphA2": "EPHA2",
    "EphA4, a receptor for epinephrine-A": "EPHA4",

    "Ephrin-B1": "EFNB1",
    "Ephrin-B2": "EFNB2",
    "Ephrin-B3": "EFNB3",

    "Fatty acid binding protein 3 (FABP3)": "FABP3",

    "Groucho family transcriptional coregulator Amino-terminal Enhancer of Split (AES)": "AES",

    "albumin": "ALB",

    "NG2": "CSPG4",
    "IL-11 receptor": "IL11RA",
    "CXCR4 receptor": "CXCR4",
    "H-2Db" : "H2-D1"
}

thp_clean["receptors_biomarker_clean"] = (
    thp_clean["receptors_biomarker_clean"]
    .replace(receptor_mapping)
)


multi_target_mapping = {

    "EGFR, EGFRvIII": [
        "EGFR",
        "EGFRvIII"
    ],

    "Neuropilin-1, Neuropilin-2": [
        "NRP1",
        "NRP2"
    ],

    "neuropilin-1, neuropilin-2": [
        "NRP1",
        "NRP2"
    ],

    "Neuropilin-1 and Neuropilin-2 (NRP1/2)": [
        "NRP1",
        "NRP2"
    ],

    "NRP-1, NRP-2": [
        "NRP1",
        "NRP2"
    ],

    "vascular endothelial growth factor receptor 2 (VEGFR2), neuropilin-1 (NRP-1)": [
        "KDR",
        "NRP1"
    ],

    "αvβ3, αvβ5": [
        "αvβ3",
        "αvβ5"
    ]
}

thp_clean["target_protein"] = thp_clean[
    "receptors_biomarker_clean"
].apply(
    lambda x: multi_target_mapping.get(x, [x])
)

# 一行多靶点 -> 多行
thp_clean = thp_clean.explode(
    "target_protein",
    ignore_index=True
)

thp_clean["target_protein"] = (
    thp_clean["target_protein"]
    .astype(str)
    .str.strip()
)


thp_clean["filter_reason"] = pd.NA


# -------------------------
# 5.1 非蛋白
# -------------------------
non_protein = {
    "Gal beta1 --> 3GalNAc alpha disaccharide",
    "Heparansulphate",
    "Heparan sulfate proteoglycans (HSPGs)(syndecans and glypicans)",
    "Sulfated Glycosaminoglycans (sulfated GAGs)",
    "human disialoganglioside GD2",
    "Hyaluronan",
    "Phosphatidylserine(PS)"
}

mask = thp_clean["target_protein"].isin(non_protein)
thp_clean.loc[mask, "filter_reason"] = "non_protein"



cell_targets = {
    "LNCaP",
    "U87R",
    "BTIC",
    "CT26",
    "-MCF 7.00"
}

mask = (  thp_clean["filter_reason"].isna() & thp_clean["target_protein"].isin(cell_targets))

thp_clean.loc[mask, "filter_reason"] = "cell_or_cell_line"


integrin_pattern = (
    r"(?i)"
    r"\bintegrins?\b"
    r"|αvβ[356]"
    r"|αVβ[356]"
    r"|α[3456]β1"
    r"|ανβ3"
    r"|avb3"
    r"|alpha.*beta"
)

mask = (
    thp_clean["filter_reason"].isna()
    & thp_clean["target_protein"].str.contains(
        integrin_pattern,
        regex=True,
        na=False
    )
)

thp_clean.loc[mask, "filter_reason"] = "integrin_complex"



variant_pattern = (
    r"(?i)"
    r"EGFRvIII"
    r"|CD44v6"
    r"|V600E"
    r"|TNC-C"
    r"|FN-EDB"
)

mask = (
    thp_clean["filter_reason"].isna()
    & thp_clean["target_protein"].str.contains(
        variant_pattern,
        regex=True,
        na=False
    )
)

thp_clean.loc[mask, "filter_reason"] = "variant_or_isoform"


vague_target = {
    "M-protein",
    "Proteoglycan",
    "Gelatinase",
    "Aminopeptidase",
    "Aminopeptidase P",

    "Neuropilins",
    "Neuropilin receptors",
    "Neuropilin (NRP)",

    "somatostatin receptor",
    "nicotine acetylcholine receptors (nAChRs)",

    "histone H1",
    "LRP Receptor",

    "GFR",
    "VGF receptor, MMP",

    "MAM, Ig domain, FNIII repeats",

    "Clotted plasma proteins",
    "Clotted-plasma proteins",

    "Fibrin-fibronectin complexes",
    "Microthrombus-associated fibrin-fibronectin complexes",
    "fibrin and fibrin-associated clotted plasma proteins (e.g. fibronectin)",

    "MMP-2-processed collagen IV",
    "Type IV collagen in basement vascular membrane",

    "Fibrinogen",
    "Fibrin",

    "NT",
    "Gastrin releasing",
    "Epinephrine-A",

    "Calpain",

    "VEGFR",

    "TNFRSF19L"
}

mask = (
    thp_clean["filter_reason"].isna()
    & thp_clean["target_protein"].isin(vague_target)
)

thp_clean.loc[mask, "filter_reason"] = "ambiguous_target"


# -------------------------
# 5.6 剩余不确定多靶点
# -------------------------
# 已经确认的多靶点在上面 explode 后已经变成单靶点。
# 如果这里还存在逗号、and、/，保守处理为不确定多靶点。
multi_pattern = r",|\band\b|/"

mask = (
    thp_clean["filter_reason"].isna()
    & thp_clean["target_protein"].str.contains(
        multi_pattern,
        regex=True,
        case=False,
        na=False
    )
)

thp_clean.loc[mask, "filter_reason"] = "uncertain_multi_target"



thp_removed = thp_clean[
    thp_clean["filter_reason"].notna()
].copy()

thp_removed.to_csv(
    os.path.join(
        output_folder,
        "thp_removed_unclear.csv"
    ),
    index=False
)

print("\nRemoved data:")
print(thp_removed["filter_reason"].value_counts())



thp_clean = thp_clean[ thp_clean["filter_reason"].isna()].copy()

thp_clean = thp_clean.drop( columns=["filter_reason"])


thp_clean = thp_clean.drop_duplicates(
    subset=[
        "sequence",
        "target_protein"
    ],
    keep="first"
).copy()


# label

thp_clean["label"] = 1


length_count = (
    thp_clean
    .drop_duplicates(subset=["sequence"])["length"]
    .value_counts()
    .sort_index()
)

length_count.to_csv(
    os.path.join(
        output_folder,
        "thp_length.csv"
    ),
    header=["count"]
)


thp_clean = thp_clean[
    [
        "id",
        "sequence",
        "target_protein",
        "length",
        "label"
    ]
].reset_index(drop=True)


thp_clean.to_csv(
    os.path.join(
        output_folder,
        "thp_clean.csv"
    ),
    index=False
)

print(thp_clean["target_protein"].isna().sum())

print("\nLength distribution:")
print(length_count)
