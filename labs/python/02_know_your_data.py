# %% [markdown]
# # Lab 02 — Know your data
#
# Every AI project starts with looking hard at the data. In this lab you explore
# three typical datasets and find the problems hiding in each:
#
# - **Part A — Field trials:** missing values, typos and outliers in agronomic data
# - **Part B — Population genetics:** quality control of a SNP genotype matrix
# - **Part C — Metagenomics:** why raw microbiome counts can't be compared directly
#
# All datasets are simulated to look like real data, problems included.

# %%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

REPO = "YOUR-GITHUB-USERNAME/ai-for-agri-bio"


def data_path(name):
    """Use the local data folder if it exists, otherwise download from GitHub."""
    for folder in ["data", "../data", "../../data"]:
        if os.path.exists(os.path.join(folder, name)):
            return os.path.join(folder, name)
    return f"https://raw.githubusercontent.com/{REPO}/main/data/{name}"


plt.rcParams["figure.figsize"] = (8, 4.5)

# %% [markdown]
# ## Part A — Field trials
#
# Each row is one plot in a multi-site variety trial: where and when it was grown,
# the variety, fertiliser, soil and weather measurements, whether disease was
# scored, and the yield.

# %%
trials = pd.read_csv(data_path("crop_trials.csv"))
print(trials.shape)
trials.head()

# %% [markdown]
# **How much is missing?** Count missing values per column.

# %%
trials.isna().sum()

# %% [markdown]
# **Do the values make sense?** Summary statistics catch impossible values fast.
# Look at the minimum and maximum of each column.

# %%
trials.describe().round(2).T

# %% [markdown]
# A yield of over 60 t/ha is impossible for a cereal, and a soil pH of 0 is a
# broken sensor. A histogram shows how rare these values are.

# %%
fig, axes = plt.subplots(1, 2)
axes[0].hist(trials["yield_t_ha"].dropna(), bins=60)
axes[0].set(title="Yield (raw)", xlabel="t/ha", ylabel="plots")
axes[1].hist(trials["soil_ph"], bins=40)
axes[1].set(title="Soil pH (raw)", xlabel="pH")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Clean, and write down every decision.** Here we treat impossible values as
# missing, then drop plots with no yield. In a real project, you would check the
# field book first: 66.5 might really be 6.65.

# %%
clean = trials.copy()
clean.loc[clean["yield_t_ha"] > 20, "yield_t_ha"] = np.nan
clean.loc[clean["soil_ph"] < 3, "soil_ph"] = np.nan
clean = clean.dropna(subset=["yield_t_ha"])
print(f"{len(trials) - len(clean)} plots removed, {len(clean)} remain")

# %% [markdown]
# **Explore relationships.** Yield differs a lot between sites — remember this,
# it becomes important in Lab 04.

# %%
order = clean.groupby("site")["yield_t_ha"].median().sort_values().index
plt.boxplot([clean.loc[clean["site"] == s, "yield_t_ha"] for s in order])
plt.xticks(range(1, len(order) + 1), order, rotation=45)
plt.title("Yield by site")
plt.ylabel("t/ha")
plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, 2)
means = clean.groupby("n_fert_kg_ha")["yield_t_ha"].mean()
axes[0].plot(means.index, means.values, marker="o")
axes[0].set(title="Nitrogen response", xlabel="N fertiliser (kg/ha)", ylabel="mean yield (t/ha)")
clean.boxplot(column="yield_t_ha", by="disease", ax=axes[1])
axes[1].set(title="Yield by disease status", xlabel="disease (0 = no, 1 = yes)")
plt.suptitle("")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Try it:** Which variety has the highest mean yield? Is it the same at every site?
# (Hint: `clean.pivot_table(values="yield_t_ha", index="variety", columns="site")`.)

# %% [markdown]
# ## Part B — Population genetics: genotype quality control
#
# The genotype matrix has one row per plant and one column per SNP. Each value
# counts copies of the alternative allele: **0** = homozygous reference,
# **1** = heterozygous, **2** = homozygous alternative. The plants come from three
# landraces, each made up of families of full siblings.

# %%
geno = pd.read_csv(data_path("genotypes.csv"), index_col="sample_id")
samples = pd.read_csv(data_path("genotype_samples.csv"), index_col="sample_id")
print(geno.shape)
geno.iloc[:5, :8]

# %%
samples.groupby("population").size()

# %% [markdown]
# **Call rate.** Missing calls cluster in bad SNPs and bad DNA samples. Remove
# SNPs and samples with more than 10% missing data.

# %%
snp_missing = geno.isna().mean(axis=0)
sample_missing = geno.isna().mean(axis=1)

fig, axes = plt.subplots(1, 2)
axes[0].hist(snp_missing, bins=50)
axes[0].axvline(0.1, color="red", ls="--")
axes[0].set(title="Missing rate per SNP", xlabel="fraction missing", ylabel="SNPs")
axes[1].hist(sample_missing, bins=50)
axes[1].axvline(0.1, color="red", ls="--")
axes[1].set(title="Missing rate per sample", xlabel="fraction missing", ylabel="samples")
plt.tight_layout()
plt.show()

# %%
geno_qc = geno.loc[sample_missing <= 0.1, snp_missing <= 0.1]
print(f"Removed {(sample_missing > 0.1).sum()} samples and {(snp_missing > 0.1).sum()} SNPs")

# %% [markdown]
# **Minor allele frequency (MAF).** SNPs where almost every plant has the same
# genotype carry no information for most analyses.

# %%
p_alt = geno_qc.mean(axis=0) / 2
maf = np.minimum(p_alt, 1 - p_alt)
plt.hist(maf, bins=50)
plt.axvline(0.05, color="red", ls="--")
plt.title("Minor allele frequency")
plt.xlabel("MAF")
plt.ylabel("SNPs")
plt.show()
geno_qc = geno_qc.loc[:, maf >= 0.05]
print(f"{geno_qc.shape[1]} SNPs pass the MAF filter")

# %% [markdown]
# **Hardy–Weinberg equilibrium (HWE).** Under random mating, a SNP with allele
# frequency p should have about 2p(1−p) heterozygotes. Too few heterozygotes can
# mean genotyping errors — but it can also mean **population structure**: pooling
# separate populations creates a heterozygote deficit (the *Wahlund effect*).
# We test each SNP in the pooled data and within each landrace.


# %%
def hwe_pvalues(g):
    """Chi-square HWE test (1 df) for every SNP column of a 0/1/2 matrix."""
    n0, n1, n2 = (g == 0).sum(), (g == 1).sum(), (g == 2).sum()
    n = n0 + n1 + n2
    p = (n1 + 2 * n2) / (2 * n)
    exp = np.vstack([n * (1 - p) ** 2, 2 * n * p * (1 - p), n * p**2])
    obs = np.vstack([n0, n1, n2])
    chi = ((obs - exp) ** 2 / exp).sum(axis=0)
    return pd.Series(stats.chi2.sf(chi, df=1), index=g.columns)


pooled = hwe_pvalues(geno_qc)
pops = samples.loc[geno_qc.index, "population"]
within = pd.concat({pop: hwe_pvalues(geno_qc[pops == pop]) for pop in pops.unique()}, axis=1)

threshold = 1e-6
n_fail = (within < threshold).sum(axis=1)
print(f"SNPs failing HWE in the pooled data: {(pooled < threshold).sum()}")
print("Number of landraces in which a SNP fails HWE:")
print(n_fail.value_counts().sort_index())

# %% [markdown]
# Pooled, many SNPs "fail" HWE because of population structure, not bad
# genotyping. Testing within populations finds the SNPs that are really broken.
# Observed versus expected heterozygosity shows the same thing:

# %%
p = geno_qc.mean() / 2
he = 2 * p * (1 - p)
ho = (geno_qc == 1).sum() / geno_qc.notna().sum()
plt.figure(figsize=(5, 5))
plt.scatter(he, ho, s=6, alpha=0.5)
plt.plot([0, 0.5], [0, 0.5], color="red", ls="--", label="HWE expectation")
plt.xlabel("expected heterozygosity (pooled)")
plt.ylabel("observed heterozygosity")
plt.legend()
plt.show()

# %%
# Families of siblings can push single SNPs out of HWE by chance, so we only
# remove SNPs that fail in at least two of the three landraces.
bad_snps = n_fail.index[n_fail >= 2]
geno_qc = geno_qc.drop(columns=bad_snps)
print(f"Final genotype matrix: {geno_qc.shape[0]} samples x {geno_qc.shape[1]} SNPs")

# %% [markdown]
# **Try it:** Change the MAF threshold to 0.01 and 0.10. How many SNPs remain?

# %% [markdown]
# ## Part C — Metagenomics: microbiome count tables
#
# Amplicon sequencing (e.g. 16S rRNA) gives a table of read counts: one row per
# soil sample, one column per ASV (amplicon sequence variant, roughly a bacterial
# taxon). Samples come from healthy and diseased fields.

# %%
counts = pd.read_csv(data_path("microbiome_counts.csv"), index_col="sample_id")
meta = pd.read_csv(data_path("microbiome_metadata.csv"), index_col="sample_id")
taxonomy = pd.read_csv(data_path("microbiome_taxonomy.csv"), index_col="taxon_id")
print(counts.shape)
counts.iloc[:5, :8]

# %% [markdown]
# **Library size.** The total number of reads per sample is set by the sequencer,
# not by the soil. A sample with 60,000 reads is not "more bacterial" than one
# with 10,000.

# %%
depth = counts.sum(axis=1)
plt.hist(depth, bins=30)
plt.title("Reads per sample (library size)")
plt.xlabel("reads")
plt.ylabel("samples")
plt.show()
print(f"Smallest library: {depth.min():,} reads; largest: {depth.max():,} reads")
print(f"Zeros in the table: {(counts == 0).mean().mean():.0%}")

# %% [markdown]
# **Relative abundance** makes samples comparable by dividing by library size.
# Here we sum ASVs to genus level and compare the top 10 genera between healthy
# and diseased fields.

# %%
rel = counts.div(depth, axis=0)
genus = rel.T.groupby(taxonomy["genus"]).sum().T
top = genus.mean().sort_values(ascending=False).index[:10]
by_status = genus[top].groupby(meta["soil_status"]).mean()
by_status["Other"] = 1 - by_status.sum(axis=1)
by_status.plot(kind="barh", stacked=True, figsize=(9, 3), colormap="tab20")
plt.xlabel("mean relative abundance")
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Compositional data.** Relative abundances sum to 1, so if one taxon goes up,
# the others must go down — even if their real numbers didn't change. The
# **centred log-ratio (CLR)** transform is a standard fix before machine learning:
# take logs (with a small pseudocount for zeros) and subtract each sample's mean.


# %%
def clr(count_table, pseudocount=0.5):
    logx = np.log(count_table + pseudocount)
    return logx.sub(logx.mean(axis=1), axis=0)


counts_clr = clr(counts)
counts_clr.iloc[:5, :6].round(2)

# %% [markdown]
# **Try it:** Which genus differs most between healthy and diseased fields?
# Compare `genus.groupby(meta["soil_status"]).mean().T` sorted by the difference.
# Then check whether the same genera differ between **sequencing batches**
# (`meta["seq_batch"]`). Keep your answer for Lab 04.
