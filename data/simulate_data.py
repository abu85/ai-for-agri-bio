"""
Simulate every dataset used in the course.

All data are SIMULATED. They are designed to look like real agricultural and
biological data and to contain the traps that real data contain (missing
values, outliers, site effects, population structure, batch effects), so the
labs can teach how to find them. Genus names in the microbiome and 16S files
are used as labels only; the DNA sequences are synthetic, not real 16S rRNA.

Run from the repository root:
    python data/simulate_data.py

The random seed is fixed, so the files are reproducible.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(2026)


# ---------------------------------------------------------------------------
# 1. Field trials (agronomy): yield and disease across sites, years, varieties
# ---------------------------------------------------------------------------
def simulate_field_trials():
    sites = {
        # name: (mean rainfall mm, mean temp C, soil pH, site yield effect t/ha)
        "North_Plain": (520, 14.0, 6.6, 0.6),
        "River_Valley": (610, 15.5, 6.9, 1.1),
        "Highland": (700, 11.5, 5.6, -0.9),
        "Coastal": (650, 16.0, 7.2, 0.3),
        "Dry_Ridge": (330, 17.5, 7.6, -1.3),
        "Lake_Shore": (580, 13.5, 6.2, 0.4),
        "Forest_Edge": (630, 12.5, 5.4, -0.5),
        "Delta": (480, 18.0, 7.0, 0.9),
    }
    years = [2021, 2022, 2023, 2024]
    year_rain = {2021: 1.0, 2022: 0.75, 2023: 1.15, 2024: 0.95}  # 2022 drought
    varieties = [f"V{i:02d}" for i in range(1, 11)]
    var_yield = dict(zip(varieties, rng.normal(0, 0.6, len(varieties))))
    var_resist = dict(zip(varieties, rng.normal(0, 1.0, len(varieties))))
    n_levels = [0, 60, 120, 180]

    rows = []
    plot = 0
    for site, (rain, temp, ph, s_eff) in sites.items():
        for year in years:
            site_year_rain = rain * year_rain[year] * rng.normal(1, 0.05)
            site_year_temp = temp + rng.normal(0, 0.6)
            for var in varieties:
                for n_rate in n_levels:
                    if rng.random() < 0.25:  # unbalanced design, like real trials
                        continue
                    plot += 1
                    p = max(5, rng.normal(25, 8))
                    k = max(40, rng.normal(160, 40))
                    om = max(0.5, rng.normal(3.0 if ph < 6.5 else 2.3, 0.6))
                    soil_ph = ph + rng.normal(0, 0.15)
                    irrig = int(site in ("Dry_Ridge", "Delta") and rng.random() < 0.6)
                    rain_eff = site_year_rain + 150 * irrig
                    humid = (rain_eff - 300) / 400 + (site_year_temp - 14) / 6
                    logit = -2.6 + 1.3 * humid - var_resist[var] + 0.004 * n_rate  # lush, high-N canopies get more disease
                    disease = int(rng.random() < 1 / (1 + np.exp(-logit)))
                    y = (
                        5.5
                        + s_eff
                        + var_yield[var]
                        + 0.022 * n_rate
                        - 0.00007 * n_rate**2
                        - 0.000012 * (rain_eff - 560) ** 2
                        - 0.9 * (soil_ph - 6.5) ** 2
                        + 0.25 * (om - 2.5)
                        + 0.004 * (p - 25)
                        - 1.4 * disease
                        + rng.normal(0, 0.55)
                    )
                    rows.append(
                        dict(
                            plot_id=f"P{plot:04d}",
                            site=site,
                            year=year,
                            variety=var,
                            n_fert_kg_ha=n_rate,
                            soil_p_mg_kg=round(p, 1),
                            soil_k_mg_kg=round(k, 0),
                            soil_ph=round(soil_ph, 2),
                            organic_matter_pct=round(om, 2),
                            rainfall_mm=round(site_year_rain, 0),
                            mean_temp_c=round(site_year_temp, 1),
                            irrigated=irrig,
                            disease=disease,
                            yield_t_ha=round(max(0.4, y), 2),
                        )
                    )
    df = pd.DataFrame(rows)

    # Real-data problems: missing values and data-entry errors
    for col, frac in [("soil_p_mg_kg", 0.05), ("organic_matter_pct", 0.04), ("soil_k_mg_kg", 0.02)]:
        idx = rng.choice(df.index, int(frac * len(df)), replace=False)
        df.loc[idx, col] = np.nan
    idx = rng.choice(df.index, 6, replace=False)
    df.loc[idx, "yield_t_ha"] = df.loc[idx, "yield_t_ha"] * 10  # decimal-point slips
    idx = rng.choice(df.index, 3, replace=False)
    df.loc[idx, "soil_ph"] = 0.0  # broken sensor
    idx = rng.choice(df.index, 8, replace=False)
    df.loc[idx, "yield_t_ha"] = np.nan  # plots lost before harvest

    df.to_csv(OUT / "crop_trials.csv", index=False)
    print(f"crop_trials.csv: {df.shape}")


# ---------------------------------------------------------------------------
# 2. Population genetics: SNP genotypes of three landraces with family structure
# ---------------------------------------------------------------------------
def simulate_genotypes():
    pops = ["Highland", "Coastal", "Valley"]
    n_fam, n_sib = 12, 10
    n_snp, n_chr = 2000, 10
    fst = 0.12

    p_anc = rng.uniform(0.05, 0.95, n_snp)
    a = p_anc * (1 - fst) / fst
    b = (1 - p_anc) * (1 - fst) / fst
    p_pop = np.vstack([rng.beta(a, b) for _ in pops])  # Balding-Nichols

    # A few SNPs that QC should remove
    mono = rng.choice(n_snp, 40, replace=False)
    p_pop[:, mono] = rng.choice([0.0, 0.995], size=(len(pops), 40))

    G, sample_rows = [], []
    sid = 0
    for k, pop in enumerate(pops):
        for f in range(n_fam):
            mother = rng.binomial(1, p_pop[k], size=(2, n_snp))
            father = rng.binomial(1, p_pop[k], size=(2, n_snp))
            for _ in range(n_sib):
                pick_m = rng.integers(0, 2, n_snp)
                pick_f = rng.integers(0, 2, n_snp)
                child = mother[pick_m, np.arange(n_snp)] + father[pick_f, np.arange(n_snp)]
                G.append(child)
                sid += 1
                sample_rows.append(dict(sample_id=f"S{sid:03d}", population=pop, family=f"{pop[:2].upper()}_F{f + 1:02d}"))
    G = np.array(G, dtype=float)

    # Trait: plant height = 300 QTL + population (environment of origin) effect + noise
    qtl = rng.choice(np.setdiff1d(np.arange(n_snp), mono), 300, replace=False)
    beta = rng.normal(0, 1, 300)
    g = G[:, qtl] @ beta
    g = (g - g.mean()) / g.std()
    pop_shift = {"Highland": -3.0, "Coastal": 2.0, "Valley": 1.0}
    shift = np.array([pop_shift[r["population"]] for r in sample_rows])
    h2 = 0.6
    height = 110 + 9 * (np.sqrt(h2) * g + np.sqrt(1 - h2) * rng.normal(0, 1, len(g))) + shift
    flowering = 62 + 0.25 * (height - 110) + rng.normal(0, 3, len(g)) - 0.4 * shift

    # Genotyping problems: a few bad SNPs (heterozygotes miscalled) and missing calls
    bad = rng.choice(np.setdiff1d(np.arange(n_snp), np.r_[mono, qtl]), 15, replace=False)
    for j in bad:
        het = np.where(G[:, j] == 1)[0]
        flip = het[rng.random(len(het)) < 0.7]
        G[flip, j] = rng.choice([0, 2], len(flip))
    G[rng.random(G.shape) < 0.01] = np.nan
    high_miss = rng.choice(n_snp, 25, replace=False)
    for j in high_miss:
        G[rng.random(G.shape[0]) < 0.3, j] = np.nan
    bad_sample = rng.choice(G.shape[0], 4, replace=False)
    for i in bad_sample:
        G[i, rng.random(n_snp) < 0.25] = np.nan

    chrom = np.repeat(np.arange(1, n_chr + 1), n_snp // n_chr)
    pos = np.concatenate([np.sort(rng.choice(np.arange(1, 50_000_000), n_snp // n_chr, replace=False)) for _ in range(n_chr)])
    snp_ids = [f"snp_{c}_{p}" for c, p in zip(chrom, pos)]
    bases = np.array(list("ACGT"))
    ref = rng.choice(bases, n_snp)
    alt = np.array([rng.choice(bases[bases != r]) for r in ref])

    geno = pd.DataFrame(G, columns=snp_ids)
    geno.insert(0, "sample_id", [r["sample_id"] for r in sample_rows])
    geno.to_csv(OUT / "genotypes.csv", index=False, na_rep="NA", float_format="%.0f")

    samples = pd.DataFrame(sample_rows)
    samples["plant_height_cm"] = np.round(height, 1)
    samples["flowering_days"] = np.round(flowering, 0).astype(int)
    samples.to_csv(OUT / "genotype_samples.csv", index=False)

    pd.DataFrame(dict(snp_id=snp_ids, chromosome=chrom, position=pos, ref=ref, alt=alt)).to_csv(
        OUT / "snp_info.csv", index=False
    )
    print(f"genotypes.csv: {geno.shape}; genotype_samples.csv: {samples.shape}")


# ---------------------------------------------------------------------------
# 3. Metagenomics: soil microbiome amplicon counts, healthy vs diseased fields
# ---------------------------------------------------------------------------
def simulate_microbiome():
    genera_pool = {
        "Proteobacteria": ["Pseudomonas", "Burkholderia", "Rhizobium", "Bradyrhizobium", "Sphingomonas",
                           "Methylobacterium", "Ralstonia", "Pectobacterium", "Xanthomonas", "Variovorax"],
        "Firmicutes": ["Bacillus", "Paenibacillus", "Clostridium"],
        "Actinobacteria": ["Streptomyces", "Arthrobacter", "Mycobacterium", "Nocardioides"],
        "Bacteroidetes": ["Flavobacterium", "Chitinophaga", "Pedobacter"],
        "Acidobacteria": ["Acidobacterium", "Bryobacter"],
        "Nitrospirae": ["Nitrospira"],
        "Verrucomicrobia": ["Chthoniobacter"],
    }
    tax = [(ph, g) for ph, gs in genera_pool.items() for g in gs]
    n_taxa = 200
    tax_rows = []
    for i in range(n_taxa):
        ph, g = tax[i % len(tax)] if i < len(tax) else tax[rng.integers(len(tax))]
        tax_rows.append(dict(taxon_id=f"ASV{i + 1:03d}", phylum=ph, genus=g))
    taxonomy = pd.DataFrame(tax_rows)

    sites = ["Field_A", "Field_B", "Field_C", "Field_D", "Field_E", "Field_F"]
    site_ph = dict(zip(sites, [5.4, 5.9, 6.3, 6.8, 7.2, 7.6]))
    n_per = 24
    meta = []
    for s in sites:
        for j in range(n_per):
            status = "healthy" if j < n_per // 2 else "diseased"
            # Batch is partly confounded with status: a classic design problem
            p_b2 = 0.25 if status == "healthy" else 0.75
            batch = "batch2" if rng.random() < p_b2 else "batch1"
            meta.append(dict(sample_id=f"M{len(meta) + 1:03d}", field=s, soil_status=status, seq_batch=batch,
                             soil_ph=round(site_ph[s] + rng.normal(0, 0.12), 2)))
    meta = pd.DataFrame(meta)

    base = rng.normal(0, 1.8, n_taxa)
    site_eff = {s: rng.normal(0, 0.6, n_taxa) for s in sites}
    ph_slope = rng.normal(0, 0.5, n_taxa)
    g = taxonomy.genus.values
    healthy_up = np.where(np.isin(g, ["Pseudomonas", "Bacillus", "Streptomyces", "Paenibacillus", "Chitinophaga"]))[0]
    diseased_up = np.where(np.isin(g, ["Ralstonia", "Pectobacterium", "Xanthomonas"]))[0]
    contam = np.where(np.isin(g, ["Methylobacterium", "Sphingomonas"]))[0]  # reagent contaminants

    counts = []
    for _, r in meta.iterrows():
        mu = base + site_eff[r.field] + ph_slope * (r.soil_ph - 6.5)
        if r.soil_status == "healthy":
            mu[healthy_up] += 0.45
        else:
            mu[diseased_up] += 0.55
        if r.seq_batch == "batch2":
            mu[contam] += 2.2
        w = np.exp(mu) * rng.gamma(2.0, 0.5, n_taxa)
        depth = int(np.clip(rng.lognormal(np.log(25000), 0.5), 3000, 90000))
        counts.append(rng.multinomial(depth, w / w.sum()))
    counts = pd.DataFrame(counts, columns=taxonomy.taxon_id)
    counts.insert(0, "sample_id", meta.sample_id)

    counts.to_csv(OUT / "microbiome_counts.csv", index=False)
    meta.to_csv(OUT / "microbiome_metadata.csv", index=False)
    taxonomy.to_csv(OUT / "microbiome_taxonomy.csv", index=False)
    print(f"microbiome_counts.csv: {counts.shape}")


# ---------------------------------------------------------------------------
# 4. RNA-seq: drought stress experiment run in two sequencing batches
# ---------------------------------------------------------------------------
def simulate_rnaseq():
    n_genes = 2000
    samples = []
    for batch in ["batch1", "batch2"]:
        for cond in ["control", "drought"]:
            for geno in ["WT", "mutant"]:
                for rep in range(3):
                    samples.append(dict(sample_id=f"R{len(samples) + 1:02d}", condition=cond, genotype=geno,
                                        seq_batch=batch, replicate=rep + 1))
    meta = pd.DataFrame(samples)
    mu = np.exp(rng.normal(4.5, 1.6, n_genes))
    lfc_cond = np.zeros(n_genes)
    lfc_cond[rng.choice(n_genes, 250, replace=False)] = rng.choice([-1, 1], 250) * rng.uniform(0.8, 2.5, 250)
    lfc_geno = np.zeros(n_genes)
    lfc_geno[rng.choice(n_genes, 80, replace=False)] = rng.choice([-1, 1], 80) * rng.uniform(0.8, 2.0, 80)
    lfc_batch = rng.normal(0, 0.9, n_genes) * (rng.random(n_genes) < 0.5)
    disp = 0.08
    X = []
    for _, r in meta.iterrows():
        sf = rng.lognormal(0, 0.25)
        m = mu * sf * 2 ** (lfc_cond * (r.condition == "drought") + lfc_geno * (r.genotype == "mutant")
                            + lfc_batch * (r.seq_batch == "batch2"))
        X.append(rng.negative_binomial(1 / disp, 1 / (1 + m * disp)))
    X = pd.DataFrame(np.array(X).T, columns=meta.sample_id)
    X.insert(0, "gene_id", [f"Gene_{i + 1:04d}" for i in range(n_genes)])
    X.to_csv(OUT / "rnaseq_counts.csv", index=False)
    meta.to_csv(OUT / "rnaseq_metadata.csv", index=False)
    print(f"rnaseq_counts.csv: {X.shape}")


# ---------------------------------------------------------------------------
# 5. Synthetic 16S-like sequences for a k-mer taxonomic classifier
# ---------------------------------------------------------------------------
def simulate_16s():
    L = 400
    bases = np.array(list("ACGT"))
    backbone = rng.choice(bases, L)
    variable = np.zeros(L, bool)
    for s, e in [(40, 90), (150, 210), (280, 340)]:
        variable[s:e] = True

    def mutate(seq, rate_var, rate_cons):
        seq = seq.copy()
        var = variable if len(seq) == L else np.zeros(len(seq), bool)
        rates = np.where(var, rate_var, rate_cons)
        hit = rng.random(len(seq)) < rates
        for i in np.where(hit)[0]:
            seq[i] = rng.choice(bases[bases != seq[i]])
        return seq

    genera = ["Bacillus", "Pseudomonas", "Streptomyces", "Rhizobium", "Flavobacterium", "Arthrobacter"]
    ref_lines, strains = [], {}
    for gname in genera:
        gseq = mutate(backbone, 0.45, 0.04)
        for sp in range(3):
            spseq = mutate(gseq, 0.08, 0.01)
            strains[(gname, sp)] = spseq
            for st in range(4):
                s = mutate(spseq, 0.01, 0.002)
                ref_lines.append(f">{gname}_sp{sp + 1}_strain{st + 1} genus={gname}\n{''.join(s)}")
    (OUT / "16s_reference.fasta").write_text("\n".join(ref_lines) + "\n")

    novel = mutate(backbone, 0.45, 0.04)  # a genus missing from the reference
    reads, truth = [], []
    read_len = 150
    keys = list(strains)
    for i in range(660):
        if i < 600:
            gname, sp = keys[rng.integers(len(keys))]
            src = mutate(strains[(gname, sp)], 0.015, 0.003)
        else:
            gname, src = "Novel_genus", mutate(novel, 0.05, 0.005)
        start = rng.integers(0, L - read_len)
        r = mutate(src[start:start + read_len], 0.005, 0.005)  # sequencing errors
        reads.append(f">read{i + 1:04d}\n{''.join(r)}")
        truth.append(dict(read_id=f"read{i + 1:04d}", true_genus=gname))
    order = rng.permutation(len(reads))
    (OUT / "16s_query_reads.fasta").write_text("\n".join(reads[i] for i in order) + "\n")
    pd.DataFrame([truth[i] for i in order]).to_csv(OUT / "16s_query_truth.csv", index=False)
    print(f"16s_reference.fasta: {len(ref_lines)} seqs; 16s_query_reads.fasta: {len(reads)} reads")


if __name__ == "__main__":
    simulate_field_trials()
    simulate_genotypes()
    simulate_microbiome()
    simulate_rnaseq()
    simulate_16s()
