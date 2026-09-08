import requests
import pandas as pd
import time


# 1. 读取配对数据


data = pd.read_csv( "/Users/bohe/Desktop/THP CODE2/clean_data/thp_pairs.csv")





# 获取每个 target 的氨基酸序列


target_sequences = {}

for target in data["target_protein"].dropna().unique():

    url = "https://rest.uniprot.org/uniprotkb/search"

    
#这里包含了人类9606，也有小鼠10096（以后决定到底怎么确定靶点的序列，最近thp的网站崩溃了点击不了）
    params = {
        "query": f"(gene_exact:{target}) AND (organism_id:9606 OR organism_id:10090) AND (reviewed:true)",
        "format": "tsv",
        "fields": "accession,gene_names,sequence",
        "size": 1
    }

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    lines = r.text.strip().split("\n")

    if len(lines) > 1:
        parts = lines[1].split("\t")
        target_sequences[target] = parts[-1]
        print(target, "OK")
    else:
        target_sequences[target] = pd.NA
        print(target, "NOT FOUND")

    time.sleep(0.1)


#  add target_sequence


data["target_sequence"] = data["target_protein"].map(target_sequences)


#  save as


data.to_csv( "/Users/bohe/Desktop/THP CODE2/clean_data/thp_pairs_sequence.csv", index=False)

print("\nMissing target sequences:")
print(data.loc[data["target_sequence"].isna(), "target_protein"].value_counts())
