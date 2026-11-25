::: {.callout-note}
This lab needs internet access (and ideally a GPU), so it is shown without outputs. Run it in Google Colab with the button above.
:::

**Run this lab in Google Colab (R runtime).** It downloads sequences from
UniProt, structures from the AlphaFold Database and a small protein language
model from Hugging Face. A GPU is not required.

Protein language models such as **ESM-2** learned the "grammar" of proteins
by reading hundreds of millions of sequences. Without being told anything
about structure or function, they place similar proteins close together. You
can use their **embeddings** (a list of numbers describing each protein) as
features for your own models.

The UniProt and AlphaFold parts are pure R. ESM-2 runs in Python's PyTorch,
which we call from R with `reticulate`.

1. Download plant protein sequences from four families
2. Turn each sequence into an ESM-2 embedding
3. Visualise and classify the families — and compare with a simple baseline
4. Read AlphaFold confidence scores (pLDDT) for a disordered and a folded protein

```r
pkgs <- c("ggplot2", "httr", "jsonlite", "reticulate", "glmnet")
new <- setdiff(pkgs, rownames(installed.packages()))
if (length(new)) install.packages(new)

library(ggplot2)
library(httr)
library(jsonlite)
library(reticulate)
library(glmnet)
options(repr.plot.width = 8, repr.plot.height = 4.5, repr.plot.res = 110)
theme_set(theme_minimal(base_size = 12))
set.seed(42)
```

### 1. Download sequences from UniProt

We take reviewed (Swiss-Prot) plant proteins from four families with different
jobs: **dehydrins** (protect cells during drought and cold), **NAC
transcription factors** (stress and development), **MADS-box transcription
factors** (flowering) and **aquaporins** (water channels).

```r
uniprot_fasta <- function(query, size = 40) {
  r <- GET("https://rest.uniprot.org/uniprotkb/search",
           query = list(query = query, format = "fasta", size = size), timeout(60))
  stop_for_status(r)
  lines <- strsplit(content(r, as = "text", encoding = "UTF-8"), "\n")[[1]]
  h <- grep("^>", lines)
  ends <- c(h[-1] - 1, length(lines))
  data.frame(accession = sapply(strsplit(lines[h], "\\|"), `[`, 2),
             sequence = mapply(function(s, e) paste(lines[(s + 1):e], collapse = ""), h, ends))
}

plants <- "(taxonomy_id:33090) AND (reviewed:true) AND (length:[80 TO 600])"
families <- c("Dehydrin" = "protein_name:dehydrin",
              "NAC TF" = 'protein_name:"NAC domain-containing protein"',
              "MADS-box TF" = 'protein_name:"MADS-box"',
              "Aquaporin" = "protein_name:aquaporin")
proteins <- do.call(rbind, lapply(names(families), function(f) {
  d <- uniprot_fasta(paste0("(", families[[f]], ") AND ", plants))
  d$family <- f
  d
}))
proteins <- proteins[!duplicated(proteins$accession), ]
proteins$length <- nchar(proteins$sequence)
aggregate(length ~ family, data = proteins, FUN = function(x) c(n = length(x), mean = round(mean(x))))
```

### 2. A simple baseline: amino-acid composition

Before reaching for a big model, try something simple. The fraction of each of
the 20 amino acids is a 20-number summary of a protein.

```r
AA <- strsplit("ACDEFGHIKLMNPQRSTVWY", "")[[1]]
composition <- t(sapply(proteins$sequence, function(s) {
  aa <- strsplit(s, "")[[1]]
  sapply(AA, function(a) mean(aa == a))
}, USE.NAMES = FALSE))
```

### 3. ESM-2 embeddings

We use the smallest ESM-2 model (8 million parameters, 6 layers). Each amino
acid gets a 320-number vector; averaging over the sequence gives one vector per
protein.

```r
if (!py_module_available("transformers") || !py_module_available("torch")) {
  py_install(c("torch", "transformers"), pip = TRUE)
}
torch <- import("torch")
transformers <- import("transformers")
model_name <- "facebook/esm2_t6_8M_UR50D"
tokenizer <- transformers$AutoTokenizer$from_pretrained(model_name)
esm <- transformers$EsmModel$from_pretrained(model_name)
esm$eval()

embed <- function(seq) {
  tokens <- tokenizer(seq, return_tensors = "pt", truncation = TRUE, max_length = 1024L)
  hidden <- esm(input_ids = tokens$input_ids, attention_mask = tokens$attention_mask)$last_hidden_state
  h <- hidden$detach()$numpy()[1, , ]  # tokens x 320
  colMeans(h[2:(nrow(h) - 1), ])       # drop start/end tokens, average
}
embeddings <- t(sapply(proteins$sequence, embed, USE.NAMES = FALSE))
dim(embeddings)
```

```r
pca_plot <- function(X, title) {
  pcs <- prcomp(X, scale. = TRUE)$x[, 1:2]
  ggplot(data.frame(pcs, family = proteins$family), aes(PC1, PC2, colour = family)) +
    geom_point(alpha = 0.8) + ggtitle(title)
}
pca_plot(composition, "Amino-acid composition")
```

```r
pca_plot(embeddings, "ESM-2 embeddings")
```

**Can a simple classifier tell the families apart?** Cross-validated accuracy
of a (lightly regularised) multinomial logistic regression on each
representation:

```r
folds <- integer(nrow(proteins))
for (f in unique(proteins$family)) {
  idx <- which(proteins$family == f)
  folds[idx] <- sample(rep(1:5, length.out = length(idx)))
}
cv_accuracy <- function(X) {
  correct <- sapply(1:5, function(k) {
    fit <- cv.glmnet(X[folds != k, ], proteins$family[folds != k], family = "multinomial", alpha = 0)
    mean(predict(fit, X[folds == k, ], s = "lambda.min", type = "class") == proteins$family[folds == k])
  })
  mean(correct)
}
cat("amino-acid composition: accuracy =", round(cv_accuracy(composition), 2), "\n")
cat("ESM-2 embeddings:       accuracy =", round(cv_accuracy(embeddings), 2), "\n")
```

Remember Lab 04: proteins from closely related species can be nearly
identical, so a random split may be optimistic. A stricter test would keep
similar sequences (e.g. > 50% identity) on the same side of the split.

### 4. AlphaFold: how confident is the predicted structure?

The AlphaFold Database holds predicted structures for most UniProt proteins.
Each residue has a confidence score, **pLDDT** (0–100), stored in the B-factor
column of the structure file:

- above 90: very high confidence
- 70–90: confident
- 50–70: low confidence
- below 50: very low — often an **intrinsically disordered** region

We compare one dehydrin with one aquaporin from our list.

```r
alphafold_plddt <- function(accession) {
  meta <- GET(paste0("https://alphafold.ebi.ac.uk/api/prediction/", accession), timeout(60))
  if (status_code(meta) != 200) return(NULL)
  pdb_url <- fromJSON(content(meta, as = "text", encoding = "UTF-8"))$pdbUrl[1]
  pdb <- readLines(pdb_url, warn = FALSE)
  ca <- pdb[startsWith(pdb, "ATOM") & trimws(substr(pdb, 13, 16)) == "CA"]
  as.numeric(substr(ca, 61, 66))
}

examples <- list()
for (f in c("Dehydrin", "Aquaporin")) {
  for (acc in proteins$accession[proteins$family == f]) {
    p <- alphafold_plddt(acc)
    if (!is.null(p)) { examples[[sprintf("%s (%s), mean %.0f", f, acc, mean(p))]] <- p; break }
  }
}
plddt_df <- do.call(rbind, lapply(names(examples), function(n)
  data.frame(protein = n, residue = seq_along(examples[[n]]), pLDDT = examples[[n]])))
options(repr.plot.width = 10, repr.plot.height = 4)
ggplot(plddt_df, aes(residue, pLDDT, colour = protein)) +
  geom_hline(yintercept = c(50, 70, 90), colour = "grey70", linetype = 3) +
  geom_line() + ylim(0, 100) +
  labs(title = "AlphaFold confidence along the sequence", colour = NULL)
```

The aquaporin, a membrane channel with a fixed shape, is predicted with high
confidence. The dehydrin is mostly low-confidence, often below 50 — **not a failure**: dehydrins are
intrinsically disordered proteins that only take shape when they bind
membranes or other molecules. Low pLDDT is itself biological information.

**Predict your own protein.** For a sequence not in the database (a new
allele, a gene from your organism), use the free
[AlphaFold Server](https://alphafoldserver.com) (AlphaFold 3, web form,
non-commercial use) or ColabFold notebooks.

**Try it:** Add a fifth family, for example `protein_name:"heat shock protein"`,
and rerun. Where does it fall in the embedding plot?
