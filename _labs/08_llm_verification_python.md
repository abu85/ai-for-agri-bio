Large language models (Claude, ChatGPT, Gemini and others) are excellent
research assistants for code, summaries and data cleaning. They are also
confidently wrong in ways that are easy to miss. The skill is not writing the
prompt — it is **checking the answer**.

The functions below are typical of what an LLM produces when asked for small
bioinformatics helpers. They look right. They run without errors. Each has a
bug. Your job: write tests with answers you can work out by hand, and find the
bugs before they find your data.

```python
import numpy as np
import pandas as pd


def check(description, got, expected, tol=1e-6):
    """Print PASS or FAIL for one test."""
    if isinstance(expected, float):
        ok = got is not None and abs(got - expected) < tol
    elif isinstance(expected, pd.DataFrame):
        ok = np.allclose(got.values, expected.values)
    else:
        ok = got == expected
    print(f"{'PASS' if ok else 'FAIL'}  {description}: got {got!r}" + ("" if ok else f", expected {expected!r}"))
```

### Function 1 — GC content

*Prompt: "Write a Python function that returns the GC content of a DNA sequence."*

```python
def gc_content(seq):
    """Return the fraction of G and C bases in a DNA sequence."""
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq)
```

**Tests.** Choose inputs where you know the answer, including awkward ones:
lowercase letters (soft-masked repeats in genome files) and `N` (unknown bases).

```python
check("simple", gc_content("GGCC"), 1.0)
check("half", gc_content("ATGC"), 0.5)
check("lowercase (soft-masked)", gc_content("atgc"), 0.5)
check("ignores N", gc_content("GCNN"), 1.0)
```

```{.text .lab-output}
PASS  simple: got 1.0
PASS  half: got 0.5
FAIL  lowercase (soft-masked): got 0.0, expected 0.5
FAIL  ignores N: got 0.5, expected 1.0
```

Two failures: lowercase bases are ignored, and `N` is counted in the length.
Whether N should count is a *scientific* decision the LLM made silently — and
it changes results for gappy assemblies. A fixed version:

```python
def gc_content_fixed(seq):
    seq = seq.upper()
    acgt = sum(seq.count(b) for b in "ACGT")
    return (seq.count("G") + seq.count("C")) / acgt if acgt else float("nan")


check("lowercase (soft-masked)", gc_content_fixed("atgc"), 0.5)
check("ignores N", gc_content_fixed("GCNN"), 1.0)
```

```{.text .lab-output}
PASS  lowercase (soft-masked): got 0.5
PASS  ignores N: got 1.0
```

### Function 2 — Reverse complement

*Prompt: "Write a function to get the reverse complement of a DNA sequence."*

```python
def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    comp = seq.replace("A", "T").replace("T", "A").replace("G", "C").replace("C", "G")
    return comp[::-1]
```

```python
check("palindrome site (EcoRI)", reverse_complement("GAATTC"), "GAATTC")
check("simple", reverse_complement("AACG"), "CGTT")
```

```{.text .lab-output}
FAIL  palindrome site (EcoRI): got 'GAAAAG', expected 'GAATTC'
FAIL  simple: got 'GGAA', expected 'CGTT'
```

The chained `replace` calls overwrite each other: every A becomes T, and then
every T (including the new ones) becomes A. A test on a **known palindrome**
such as the EcoRI site catches it instantly.

```python
def reverse_complement_fixed(seq):
    return seq.translate(str.maketrans("ACGTacgtN", "TGCAtgcaN"))[::-1]


check("palindrome site (EcoRI)", reverse_complement_fixed("GAATTC"), "GAATTC")
check("simple", reverse_complement_fixed("AACG"), "CGTT")
```

```{.text .lab-output}
PASS  palindrome site (EcoRI): got 'GAATTC'
PASS  simple: got 'CGTT'
```

### Function 3 — Allele frequency (population genetics)

*Prompt: "Given counts of genotypes AA, Aa and aa, compute the frequency of
allele A and the expected heterozygosity under Hardy–Weinberg."*

```python
def allele_freq_and_he(n_AA, n_Aa, n_aa):
    """Return frequency of allele A and expected heterozygosity 2pq."""
    n = n_AA + n_Aa + n_aa
    p = (n_AA + n_Aa) / n
    q = 1 - p
    return p, 2 * p * q
```

**Test with a case you can do by hand.** 25 AA, 50 Aa, 25 aa: there are 200
alleles, 100 of them A, so p = 0.5 and expected heterozygosity = 0.5.

```python
p, he = allele_freq_and_he(25, 50, 25)
check("p for 25/50/25", p, 0.5)
check("He for 25/50/25", he, 0.5)
p, he = allele_freq_and_he(0, 100, 0)
check("p when all heterozygous", p, 0.5)
```

```{.text .lab-output}
FAIL  p for 25/50/25: got 0.75, expected 0.5
FAIL  He for 25/50/25: got 0.375, expected 0.5
FAIL  p when all heterozygous: got 1.0, expected 0.5
```

The function counts *individuals* carrying A, not *alleles*. Each AA plant has
two copies and each Aa plant one: p = (2·AA + Aa) / 2n. It gives 0.75 instead
of 0.5 — a plausible-looking number that would quietly distort every
downstream analysis.

```python
def allele_freq_and_he_fixed(n_AA, n_Aa, n_aa):
    n = n_AA + n_Aa + n_aa
    p = (2 * n_AA + n_Aa) / (2 * n)
    return p, 2 * p * (1 - p)


p, he = allele_freq_and_he_fixed(25, 50, 25)
check("p for 25/50/25", p, 0.5)
check("He for 25/50/25", he, 0.5)
```

```{.text .lab-output}
PASS  p for 25/50/25: got 0.5
PASS  He for 25/50/25: got 0.5
```

### Function 4 — Relative abundance (metagenomics)

*Prompt: "Convert my OTU count table (samples as rows, taxa as columns) to
relative abundances."*

```python
def relative_abundance(counts):
    """Convert a count table to relative abundances."""
    return counts / counts.sum()
```

**Test the defining property:** after conversion, every *sample* (row) should
sum to 1. Use a tiny table you can check by eye.

```python
toy = pd.DataFrame({"Bacillus": [10, 0], "Pseudomonas": [30, 5], "Streptomyces": [60, 15]},
                   index=["soil_1", "soil_2"])
rel = relative_abundance(toy)
print(rel.round(3))
check("rows sum to 1", rel.sum(axis=1).round(6).tolist(), [1.0, 1.0])
```

```{.text .lab-output}
        Bacillus  Pseudomonas  Streptomyces
soil_1       1.0        0.857           0.8
soil_2       0.0        0.143           0.2
FAIL  rows sum to 1: got [2.657143, 0.342857], expected [1.0, 1.0]
```

`counts.sum()` sums each *column* (taxon), so the function divides by taxon
totals: the classic axis bug. The output is still a table of numbers between
0 and 1, so nothing looks wrong.

```python
def relative_abundance_fixed(counts):
    return counts.div(counts.sum(axis=1), axis=0)


check("rows sum to 1", relative_abundance_fixed(toy).sum(axis=1).round(6).tolist(), [1.0, 1.0])
```

```{.text .lab-output}
PASS  rows sum to 1: got [1.0, 1.0]
```

### What to take away

**A verification checklist for LLM output**

1. **Test with known answers**, including edge cases: empty input, lowercase,
   missing values, a palindrome, a single sample.
2. **Check defining properties**: proportions sum to 1, a reverse complement
   applied twice returns the original, probabilities lie between 0 and 1.
3. **Ask what decisions were made silently** (how are N's, missing values or
   ties handled?) and whether they are right for *your* data.
4. **Compare with an established tool** on a small example (Biopython,
   `vegan`, PLINK, scikit-bio).
5. **Never trust a citation** you have not opened yourself. LLMs can invent
   plausible papers, authors and DOIs.

**Prompting tips that help**

- Give context: the data format, a few example rows, the language and packages.
- Ask for tests together with the code: *"Also write three tests with inputs
  where the answer can be checked by hand."*
- Ask the model to list its assumptions.
- Ask it to explain the code line by line — then check the explanation.

**Data privacy:** do not paste unpublished data, patient or farmer information,
or confidential breeding data into public chatbots. Use your institution's
approved tools, a local model, or synthetic examples with the same structure.

**Try it:** Ask an LLM to write a function that computes the Shannon diversity
of each sample in a count table. Write at least three tests before you run it.
(Hint: a sample with 4 equally common taxa has Shannon index ln 4 ≈ 1.386.)
