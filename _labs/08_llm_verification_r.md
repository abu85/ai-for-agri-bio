Large language models (Claude, ChatGPT, Gemini and others) are excellent
research assistants for code, summaries and data cleaning. They are also
confidently wrong in ways that are easy to miss. The skill is not writing the
prompt — it is **checking the answer**.

The functions below are typical of what an LLM produces when asked for small
bioinformatics helpers. They look right. They run without errors. Each has a
bug. Your job: write tests with answers you can work out by hand, and find the
bugs before they find your data.

```r
# Print PASS or FAIL for one test
check <- function(description, got, expected, tol = 1e-6) {
  ok <- if (is.numeric(expected)) isTRUE(all(abs(got - expected) < tol)) else identical(got, expected)
  msg <- paste0(if (ok) "PASS" else "FAIL", "  ", description, ": got ", paste(format(got), collapse = ", "))
  if (!ok) msg <- paste0(msg, ", expected ", paste(format(expected), collapse = ", "))
  cat(msg, "\n")
}
```

### Function 1 — GC content

*Prompt: "Write an R function that returns the GC content of a DNA sequence."*

```r
gc_content <- function(seq) {
  bases <- strsplit(seq, "")[[1]]
  sum(bases %in% c("G", "C")) / length(bases)
}
```

**Tests.** Choose inputs where you know the answer, including awkward ones:
lowercase letters (soft-masked repeats in genome files) and `N` (unknown bases).

```r
check("simple", gc_content("GGCC"), 1.0)
check("half", gc_content("ATGC"), 0.5)
check("lowercase (soft-masked)", gc_content("atgc"), 0.5)
check("ignores N", gc_content("GCNN"), 1.0)
```

```{.text .lab-output}
PASS  simple: got 1 
```

```{.text .lab-output}
PASS  half: got 0.5 
```

```{.text .lab-output}
FAIL  lowercase (soft-masked): got 0, expected 0.5 
```

```{.text .lab-output}
FAIL  ignores N: got 0.5, expected 1 
```

Two failures: lowercase bases are ignored, and `N` is counted in the length.
Whether N should count is a *scientific* decision the LLM made silently — and
it changes results for gappy assemblies. A fixed version:

```r
gc_content_fixed <- function(seq) {
  bases <- strsplit(toupper(seq), "")[[1]]
  acgt <- sum(bases %in% c("A", "C", "G", "T"))
  if (acgt == 0) return(NA_real_)
  sum(bases %in% c("G", "C")) / acgt
}
check("lowercase (soft-masked)", gc_content_fixed("atgc"), 0.5)
check("ignores N", gc_content_fixed("GCNN"), 1.0)
```

```{.text .lab-output}
PASS  lowercase (soft-masked): got 0.5 
```

```{.text .lab-output}
PASS  ignores N: got 1 
```

### Function 2 — Reverse complement

*Prompt: "Write a function to get the reverse complement of a DNA sequence."*

```r
reverse_complement <- function(seq) {
  comp <- gsub("A", "T", seq)
  comp <- gsub("T", "A", comp)
  comp <- gsub("G", "C", comp)
  comp <- gsub("C", "G", comp)
  paste(rev(strsplit(comp, "")[[1]]), collapse = "")
}
```

```r
check("palindrome site (EcoRI)", reverse_complement("GAATTC"), "GAATTC")
check("simple", reverse_complement("AACG"), "CGTT")
```

```{.text .lab-output}
FAIL  palindrome site (EcoRI): got GAAAAG, expected GAATTC 
```

```{.text .lab-output}
FAIL  simple: got GGAA, expected CGTT 
```

The sequential substitutions overwrite each other: every A becomes T, and then
every T (including the new ones) becomes A. A test on a **known palindrome**
such as the EcoRI site catches it instantly. `chartr` swaps all letters at once:

```r
reverse_complement_fixed <- function(seq) {
  paste(rev(strsplit(chartr("ACGTacgt", "TGCAtgca", seq), "")[[1]]), collapse = "")
}
check("palindrome site (EcoRI)", reverse_complement_fixed("GAATTC"), "GAATTC")
check("simple", reverse_complement_fixed("AACG"), "CGTT")
```

```{.text .lab-output}
PASS  palindrome site (EcoRI): got GAATTC 
```

```{.text .lab-output}
PASS  simple: got CGTT 
```

### Function 3 — Allele frequency (population genetics)

*Prompt: "Given counts of genotypes AA, Aa and aa, compute the frequency of
allele A and the expected heterozygosity under Hardy–Weinberg."*

```r
allele_freq_and_he <- function(n_AA, n_Aa, n_aa) {
  n <- n_AA + n_Aa + n_aa
  p <- (n_AA + n_Aa) / n
  q <- 1 - p
  c(p = p, He = 2 * p * q)
}
```

**Test with a case you can do by hand.** 25 AA, 50 Aa, 25 aa: there are 200
alleles, 100 of them A, so p = 0.5 and expected heterozygosity = 0.5.

```r
res <- allele_freq_and_he(25, 50, 25)
check("p for 25/50/25", unname(res["p"]), 0.5)
check("He for 25/50/25", unname(res["He"]), 0.5)
check("p when all heterozygous", unname(allele_freq_and_he(0, 100, 0)["p"]), 0.5)
```

```{.text .lab-output}
FAIL  p for 25/50/25: got 0.75, expected 0.5 
```

```{.text .lab-output}
FAIL  He for 25/50/25: got 0.375, expected 0.5 
```

```{.text .lab-output}
FAIL  p when all heterozygous: got 1, expected 0.5 
```

The function counts *individuals* carrying A, not *alleles*. Each AA plant has
two copies and each Aa plant one: p = (2·AA + Aa) / 2n. It gives 0.75 instead
of 0.5 — a plausible-looking number that would quietly distort every
downstream analysis.

```r
allele_freq_and_he_fixed <- function(n_AA, n_Aa, n_aa) {
  n <- n_AA + n_Aa + n_aa
  p <- (2 * n_AA + n_Aa) / (2 * n)
  c(p = p, He = 2 * p * (1 - p))
}
res <- allele_freq_and_he_fixed(25, 50, 25)
check("p for 25/50/25", unname(res["p"]), 0.5)
check("He for 25/50/25", unname(res["He"]), 0.5)
```

```{.text .lab-output}
PASS  p for 25/50/25: got 0.5 
```

```{.text .lab-output}
PASS  He for 25/50/25: got 0.5 
```

### Function 4 — Relative abundance (metagenomics)

*Prompt: "Convert my OTU count table (samples as rows, taxa as columns) to
relative abundances."*

```r
relative_abundance <- function(counts) {
  counts / colSums(counts)
}
```

**Test the defining property:** after conversion, every *sample* (row) should
sum to 1. Use a tiny table you can check by eye.

```r
toy <- matrix(c(10, 0, 30, 5, 60, 15), nrow = 2,
              dimnames = list(c("soil_1", "soil_2"), c("Bacillus", "Pseudomonas", "Streptomyces")))
rel <- relative_abundance(toy)
round(rel, 3)
check("rows sum to 1", unname(rowSums(rel)), c(1, 1))
```

::: {.lab-table}
| <!--/--> | Bacillus | Pseudomonas | Streptomyces |
|---|---|---|---|
| soil_1 | 1 | 0.4 | 1.714 |
| soil_2 | 0 | 0.5 | 0.200 |
:::

```{.text .lab-output}
FAIL  rows sum to 1: got 3.114286, 0.700000, expected 1, 1 
```

`colSums` gives one total per *taxon*, and R silently recycles that vector down
the columns, so each value is divided by the wrong total. The output is still a
table of small positive numbers, so nothing looks wrong. Dividing by `rowSums`
works because R recycles down the rows of each column; `sweep` or
`proportions(counts, 1)` make the intent explicit.

```r
relative_abundance_fixed <- function(counts) {
  sweep(counts, 1, rowSums(counts), "/")
}
check("rows sum to 1", unname(rowSums(relative_abundance_fixed(toy))), c(1, 1))
```

```{.text .lab-output}
PASS  rows sum to 1: got 1, 1 
```

### What to take away

**A verification checklist for LLM output**

1. **Test with known answers**, including edge cases: empty input, lowercase,
   missing values, a palindrome, a single sample.
2. **Check defining properties**: proportions sum to 1, a reverse complement
   applied twice returns the original, probabilities lie between 0 and 1.
3. **Ask what decisions were made silently** (how are N's, missing values or
   ties handled?) and whether they are right for *your* data.
4. **Compare with an established tool** on a small example (Bioconductor's
   `Biostrings`, `vegan`, PLINK).
5. **Never trust a citation** you have not opened yourself. LLMs can invent
   plausible papers, authors and DOIs.

**Prompting tips that help**

- Give context: the data format, a few example rows, the language and packages.
- Ask for tests together with the code: *"Also write three tests with inputs
  where the answer can be checked by hand."* (In R, the `testthat` package
  makes tests easy to keep.)
- Ask the model to list its assumptions.
- Ask it to explain the code line by line — then check the explanation.

**Data privacy:** do not paste unpublished data, patient or farmer information,
or confidential breeding data into public chatbots. Use your institution's
approved tools, a local model, or synthetic examples with the same structure.

**Try it:** Ask an LLM to write a function that computes the Shannon diversity
of each sample in a count table. Write at least three tests before you run it.
(Hint: a sample with 4 equally common taxa has Shannon index log(4) ≈ 1.386.)
