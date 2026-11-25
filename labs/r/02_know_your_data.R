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
pkgs <- c("ggplot2")
new <- setdiff(pkgs, rownames(installed.packages()))
if (length(new)) install.packages(new)

library(ggplot2)
options(repr.plot.width = 8, repr.plot.height = 4.5, repr.plot.res = 110)
theme_set(theme_minimal(base_size = 12))

REPO <- "YOUR-GITHUB-USERNAME/ai-for-agri-bio"

# Use the local data folder if it exists, otherwise download from GitHub
data_path <- function(name) {
  for (folder in c("data", "../data", "../../data")) {
    f <- file.path(folder, name)
    if (file.exists(f)) return(f)
  }
  paste0("https://raw.githubusercontent.com/", REPO, "/main/data/", name)
}

# %% [markdown]
# ## Part A — Field trials
#
# Each row is one plot in a multi-site variety trial: where and when it was grown,
# the variety, fertiliser, soil and weather measurements, whether disease was
# scored, and the yield.

# %%
trials <- read.csv(data_path("crop_trials.csv"))
dim(trials)
head(trials)

# %% [markdown]
# **How much is missing?** Count missing values per column.

# %%
colSums(is.na(trials))

# %% [markdown]
# **Do the values make sense?** Summary statistics catch impossible values fast.
# Look at the minimum and maximum of each column.

# %%
summary(trials[, c("n_fert_kg_ha", "soil_p_mg_kg", "soil_ph", "rainfall_mm", "yield_t_ha")])

# %% [markdown]
# A yield of over 60 t/ha is impossible for a cereal, and a soil pH of 0 is a
# broken sensor. A histogram shows how rare these values are.

# %%
ggplot(trials, aes(yield_t_ha)) +
  geom_histogram(bins = 60) +
  labs(title = "Yield (raw)", x = "t/ha", y = "plots")

# %% [markdown]
# **Clean, and write down every decision.** Here we treat impossible values as
# missing, then drop plots with no yield. In a real project, you would check the
# field book first: 66.5 might really be 6.65.

# %%
clean <- trials
clean$yield_t_ha[clean$yield_t_ha > 20] <- NA
clean$soil_ph[clean$soil_ph < 3] <- NA
clean <- clean[!is.na(clean$yield_t_ha), ]
cat(nrow(trials) - nrow(clean), "plots removed,", nrow(clean), "remain\n")

# %% [markdown]
# **Explore relationships.** Yield differs a lot between sites — remember this,
# it becomes important in Lab 04.

# %%
ggplot(clean, aes(reorder(site, yield_t_ha, FUN = median), yield_t_ha)) +
  geom_boxplot() +
  labs(title = "Yield by site", x = NULL, y = "t/ha") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# %%
n_means <- aggregate(yield_t_ha ~ n_fert_kg_ha, data = clean, FUN = mean)
ggplot(n_means, aes(n_fert_kg_ha, yield_t_ha)) +
  geom_line() + geom_point(size = 2) +
  labs(title = "Nitrogen response", x = "N fertiliser (kg/ha)", y = "mean yield (t/ha)")

# %%
ggplot(clean, aes(factor(disease, labels = c("no", "yes")), yield_t_ha)) +
  geom_boxplot() +
  labs(title = "Yield by disease status", x = "disease", y = "t/ha")

# %% [markdown]
# **Try it:** Which variety has the highest mean yield? Is it the same at every site?
# (Hint: `tapply(clean$yield_t_ha, list(clean$variety, clean$site), mean)`.)

# %% [markdown]
# ## Part B — Population genetics: genotype quality control
#
# The genotype matrix has one row per plant and one column per SNP. Each value
# counts copies of the alternative allele: **0** = homozygous reference,
# **1** = heterozygous, **2** = homozygous alternative. The plants come from three
# landraces, each made up of families of full siblings.

# %%
geno <- as.matrix(read.csv(data_path("genotypes.csv"), row.names = "sample_id"))
samples <- read.csv(data_path("genotype_samples.csv"), row.names = "sample_id")
dim(geno)
geno[1:5, 1:6]

# %%
table(samples$population)

# %% [markdown]
# **Call rate.** Missing calls cluster in bad SNPs and bad DNA samples. Remove
# SNPs and samples with more than 10% missing data.

# %%
snp_missing <- colMeans(is.na(geno))
sample_missing <- rowMeans(is.na(geno))

par(mfrow = c(1, 2))
hist(snp_missing, breaks = 50, main = "Missing rate per SNP", xlab = "fraction missing")
abline(v = 0.1, col = "red", lty = 2)
hist(sample_missing, breaks = 50, main = "Missing rate per sample", xlab = "fraction missing")
abline(v = 0.1, col = "red", lty = 2)
par(mfrow = c(1, 1))

# %%
geno_qc <- geno[sample_missing <= 0.1, snp_missing <= 0.1]
cat("Removed", sum(sample_missing > 0.1), "samples and", sum(snp_missing > 0.1), "SNPs\n")

# %% [markdown]
# **Minor allele frequency (MAF).** SNPs where almost every plant has the same
# genotype carry no information for most analyses.

# %%
p_alt <- colMeans(geno_qc, na.rm = TRUE) / 2
maf <- pmin(p_alt, 1 - p_alt)
hist(maf, breaks = 50, main = "Minor allele frequency", xlab = "MAF")
abline(v = 0.05, col = "red", lty = 2)
geno_qc <- geno_qc[, maf >= 0.05]
cat(ncol(geno_qc), "SNPs pass the MAF filter\n")

# %% [markdown]
# **Hardy–Weinberg equilibrium (HWE).** Under random mating, a SNP with allele
# frequency p should have about 2p(1−p) heterozygotes. Too few heterozygotes can
# mean genotyping errors — but it can also mean **population structure**: pooling
# separate populations creates a heterozygote deficit (the *Wahlund effect*).
# We test each SNP in the pooled data and within each landrace.

# %%
# Chi-square HWE test (1 df) for every SNP column of a 0/1/2 matrix
hwe_pvalues <- function(g) {
  n0 <- colSums(g == 0, na.rm = TRUE)
  n1 <- colSums(g == 1, na.rm = TRUE)
  n2 <- colSums(g == 2, na.rm = TRUE)
  n <- n0 + n1 + n2
  p <- (n1 + 2 * n2) / (2 * n)
  e0 <- n * (1 - p)^2; e1 <- 2 * n * p * (1 - p); e2 <- n * p^2
  chi <- (n0 - e0)^2 / e0 + (n1 - e1)^2 / e1 + (n2 - e2)^2 / e2
  pchisq(chi, df = 1, lower.tail = FALSE)
}

pooled <- hwe_pvalues(geno_qc)
pops <- samples[rownames(geno_qc), "population"]
within <- sapply(unique(pops), function(pop) hwe_pvalues(geno_qc[pops == pop, ]))

threshold <- 1e-6
n_fail <- rowSums(within < threshold, na.rm = TRUE)  # NA = SNP monomorphic in that landrace
cat("SNPs failing HWE in the pooled data:", sum(pooled < threshold, na.rm = TRUE), "\n")
cat("Number of landraces in which a SNP fails HWE:\n")
table(n_fail)

# %% [markdown]
# Pooled, many SNPs "fail" HWE because of population structure, not bad
# genotyping. Testing within populations finds the SNPs that are really broken.
# Observed versus expected heterozygosity shows the same thing:

# %%
p <- colMeans(geno_qc, na.rm = TRUE) / 2
het <- data.frame(expected = 2 * p * (1 - p),
                  observed = colSums(geno_qc == 1, na.rm = TRUE) / colSums(!is.na(geno_qc)))
options(repr.plot.width = 5, repr.plot.height = 5)
ggplot(het, aes(expected, observed)) +
  geom_point(alpha = 0.4, size = 0.8) +
  geom_abline(colour = "red", linetype = 2) +
  labs(x = "expected heterozygosity (pooled)", y = "observed heterozygosity")
options(repr.plot.width = 8, repr.plot.height = 4.5)

# %%
# Families of siblings can push single SNPs out of HWE by chance, so we only
# remove SNPs that fail in at least two of the three landraces.
geno_qc <- geno_qc[, n_fail < 2]
cat("Final genotype matrix:", nrow(geno_qc), "samples x", ncol(geno_qc), "SNPs\n")

# %% [markdown]
# **Try it:** Change the MAF threshold to 0.01 and 0.10. How many SNPs remain?

# %% [markdown]
# ## Part C — Metagenomics: microbiome count tables
#
# Amplicon sequencing (e.g. 16S rRNA) gives a table of read counts: one row per
# soil sample, one column per ASV (amplicon sequence variant, roughly a bacterial
# taxon). Samples come from healthy and diseased fields.

# %%
counts <- as.matrix(read.csv(data_path("microbiome_counts.csv"), row.names = "sample_id"))
meta <- read.csv(data_path("microbiome_metadata.csv"), row.names = "sample_id")
taxonomy <- read.csv(data_path("microbiome_taxonomy.csv"), row.names = "taxon_id")
dim(counts)
counts[1:5, 1:6]

# %% [markdown]
# **Library size.** The total number of reads per sample is set by the sequencer,
# not by the soil. A sample with 60,000 reads is not "more bacterial" than one
# with 10,000.

# %%
depth <- rowSums(counts)
hist(depth, breaks = 30, main = "Reads per sample (library size)", xlab = "reads")
cat("Smallest library:", min(depth), "reads; largest:", max(depth), "reads\n")
cat("Zeros in the table:", round(100 * mean(counts == 0)), "%\n")

# %% [markdown]
# **Relative abundance** makes samples comparable by dividing by library size.
# Here we sum ASVs to genus level and compare the top 10 genera between healthy
# and diseased fields.

# %%
rel <- counts / depth
genus <- t(rowsum(t(rel), taxonomy[colnames(rel), "genus"]))
top <- names(sort(colMeans(genus), decreasing = TRUE))[1:10]
by_status <- aggregate(genus[, top], by = list(status = meta[rownames(genus), "soil_status"]), FUN = mean)
long <- data.frame(status = rep(by_status$status, times = length(top)),
                   genus = rep(top, each = nrow(by_status)),
                   abundance = unlist(by_status[, top]))
options(repr.plot.height = 3)
ggplot(long, aes(abundance, status, fill = genus)) +
  geom_col() +
  labs(x = "mean relative abundance (top 10 genera)", y = NULL)
options(repr.plot.height = 4.5)

# %% [markdown]
# **Compositional data.** Relative abundances sum to 1, so if one taxon goes up,
# the others must go down — even if their real numbers didn't change. The
# **centred log-ratio (CLR)** transform is a standard fix before machine learning:
# take logs (with a small pseudocount for zeros) and subtract each sample's mean.

# %%
clr <- function(x, pseudocount = 0.5) {
  logx <- log(x + pseudocount)
  logx - rowMeans(logx)
}
counts_clr <- clr(counts)
round(counts_clr[1:5, 1:6], 2)

# %% [markdown]
# **Try it:** Which genus differs most between healthy and diseased fields?
# Compare the genus means by `meta$soil_status`, sorted by the difference.
# Then check whether the same genera differ between **sequencing batches**
# (`meta$seq_batch`). Keep your answer for Lab 04.
