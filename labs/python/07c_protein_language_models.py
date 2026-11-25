# %% [markdown]
# # Lab 07C — Protein language models and AlphaFold
#
# **Run this lab in Google Colab.** It downloads sequences from UniProt,
# structures from the AlphaFold Database and a small protein language model
# from Hugging Face. A GPU is not required (the model is small).
#
# Protein language models such as **ESM-2** learned the "grammar" of proteins
# by reading hundreds of millions of sequences. Without being told anything
# about structure or function, they place similar proteins close together. You
# can use their **embeddings** (a list of numbers describing each protein) as
# features for your own models — no training from scratch.
#
# 1. Download plant protein sequences from four families
# 2. Turn each sequence into an ESM-2 embedding
# 3. Visualise and classify the families — and compare with a simple baseline
# 4. Read AlphaFold confidence scores (pLDDT) for a disordered and a folded protein

# %%
import io
import subprocess
import sys

try:
    import transformers  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers"], check=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import torch
from transformers import AutoTokenizer, EsmModel
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SEED = 42

# %% [markdown]
# ## 1. Download sequences from UniProt
#
# We take reviewed (Swiss-Prot) plant proteins from four families with different
# jobs: **dehydrins** (protect cells during drought and cold), **NAC
# transcription factors** (stress and development), **MADS-box transcription
# factors** (flowering) and **aquaporins** (water channels).

# %%
def uniprot_fasta(query, size=40):
    """Search UniProt and return a list of (accession, sequence)."""
    r = requests.get("https://rest.uniprot.org/uniprotkb/search",
                     params={"query": query, "format": "fasta", "size": size}, timeout=60)
    r.raise_for_status()
    records, acc, seq = [], None, []
    for line in io.StringIO(r.text):
        line = line.strip()
        if line.startswith(">"):
            if acc:
                records.append((acc, "".join(seq)))
            acc, seq = line.split("|")[1], []
        elif line:
            seq.append(line)
    if acc:
        records.append((acc, "".join(seq)))
    return records


plants = "(taxonomy_id:33090) AND (reviewed:true) AND (length:[80 TO 600])"
families = {
    "Dehydrin": "protein_name:dehydrin",
    "NAC TF": 'protein_name:"NAC domain-containing protein"',
    "MADS-box TF": 'protein_name:"MADS-box"',
    "Aquaporin": "protein_name:aquaporin",
}
rows = []
for fam, q in families.items():
    for acc, seq in uniprot_fasta(f"({q}) AND {plants}"):
        rows.append(dict(accession=acc, family=fam, sequence=seq, length=len(seq)))
proteins = pd.DataFrame(rows).drop_duplicates("accession")
proteins.groupby("family")["length"].describe()[["count", "mean", "min", "max"]]

# %% [markdown]
# ## 2. A simple baseline: amino-acid composition
#
# Before reaching for a big model, try something simple. The fraction of each of
# the 20 amino acids is a 20-number summary of a protein.

# %%
AA = "ACDEFGHIKLMNPQRSTVWY"
composition = np.array([[s.count(a) / len(s) for a in AA] for s in proteins["sequence"]])

# %% [markdown]
# ## 3. ESM-2 embeddings
#
# We use the smallest ESM-2 model (8 million parameters, 6 layers). Each amino
# acid gets a 320-number vector; averaging over the sequence gives one vector per
# protein.

# %%
model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
esm = EsmModel.from_pretrained(model_name).eval()


@torch.no_grad()
def embed(seq):
    tokens = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
    hidden = esm(**tokens).last_hidden_state[0]  # (tokens, 320)
    return hidden[1:-1].mean(dim=0).numpy()     # drop start/end tokens, average


embeddings = np.vstack([embed(s) for s in proteins["sequence"]])
print(embeddings.shape)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (X, title) in zip(axes, [(composition, "Amino-acid composition"), (embeddings, "ESM-2 embeddings")]):
    xy = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
    for fam in families:
        m = (proteins["family"] == fam).values
        ax.scatter(xy[m, 0], xy[m, 1], label=fam, alpha=0.75)
    ax.set(title=title, xlabel="PC1", ylabel="PC2")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Can a simple classifier tell the families apart?** Cross-validated accuracy
# with logistic regression on each representation:

# %%
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
for X, name in [(composition, "amino-acid composition"), (embeddings, "ESM-2 embeddings")]:
    acc = cross_val_score(clf, X, proteins["family"], cv=cv).mean()
    print(f"{name:25s} accuracy = {acc:.2f}")

# %% [markdown]
# Remember Lab 04: proteins from closely related species can be nearly
# identical, so a random split may be optimistic. A stricter test would keep
# similar sequences (e.g. > 50% identity) on the same side of the split.
#
# ## 4. AlphaFold: how confident is the predicted structure?
#
# The AlphaFold Database holds predicted structures for most UniProt proteins.
# Each residue has a confidence score, **pLDDT** (0–100), stored in the B-factor
# column of the structure file:
#
# - above 90: very high confidence
# - 70–90: confident
# - 50–70: low confidence
# - below 50: very low — often an **intrinsically disordered** region
#
# We compare one dehydrin with one aquaporin from our list.

# %%
def alphafold_plddt(accession):
    """Download the AlphaFold model for a UniProt accession and return per-residue pLDDT."""
    meta = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{accession}", timeout=60)
    if meta.status_code != 200:
        return None
    pdb = requests.get(meta.json()[0]["pdbUrl"], timeout=60).text
    return np.array([float(line[60:66]) for line in pdb.splitlines()
                     if line.startswith("ATOM") and line[12:16].strip() == "CA"])


examples = {}
for fam in ["Dehydrin", "Aquaporin"]:
    for acc in proteins.loc[proteins["family"] == fam, "accession"]:
        plddt = alphafold_plddt(acc)
        if plddt is not None:
            examples[f"{fam} ({acc})"] = plddt
            break

plt.figure(figsize=(10, 4))
for name, plddt in examples.items():
    plt.plot(np.arange(1, len(plddt) + 1), plddt, label=f"{name}, mean {plddt.mean():.0f}")
for y in (50, 70, 90):
    plt.axhline(y, color="gray", ls=":", lw=1)
plt.ylim(0, 100)
plt.xlabel("residue")
plt.ylabel("pLDDT")
plt.legend()
plt.title("AlphaFold confidence along the sequence")
plt.show()

# %% [markdown]
# The aquaporin, a membrane channel with a fixed shape, is predicted with high
# confidence. The dehydrin is mostly low-confidence, often below 50 — **not a failure**: dehydrins are
# intrinsically disordered proteins that only take shape when they bind
# membranes or other molecules. Low pLDDT is itself biological information.
#
# **Predict your own protein.** For a sequence not in the database (a new
# allele, a gene from your organism), use the free
# [AlphaFold Server](https://alphafoldserver.com) (AlphaFold 3, web form,
# non-commercial use) or ColabFold notebooks.
#
# **Try it:** Add a fifth family, for example `protein_name:"heat shock protein"`,
# and rerun. Where does it fall in the embedding plot?
