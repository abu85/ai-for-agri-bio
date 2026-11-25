# AI for Agriculturists & Biologists

A two-day, hands-on workshop that takes agricultural and biological researchers from "what is AI?" to training, testing and critically judging machine-learning models — with **every lab in both Python and R**, and examples from agronomy, **population genetics** and **metagenomics**.

**Website:** https://abu85.github.io/ai-for-agri-bio/

## What's inside

| Module | Topic | Lab |
|---|---|---|
| 1 | What AI is (and isn't) | — |
| 2 | Data is the foundation | Field-trial cleaning · SNP genotype QC and HWE · microbiome counts and CLR |
| 3 | Your first models | Yield and disease prediction · genomic prediction (ridge/rrBLUP vs random forest) · soil-health classifier |
| 4 | Evaluating models honestly | Overfitting · site leakage · relatives and population structure · batch confounding · feature-selection leakage |
| 5 | Finding structure | RNA-seq PCA and batch effects · SNP PCA, k-means, UMAP · Bray–Curtis PCoA and PERMANOVA |
| 6 | Neural networks | One neuron by hand · training curves · regularisation · comparison with random forest |
| 7 | Images, sequences, proteins | 7A plant-disease transfer learning · 7B k-mer read classification · 7C ESM-2 embeddings and AlphaFold pLDDT |
| 8 | LLMs as research assistants | Find the bugs in LLM-written bioinformatics code |
| 9 | Responsible AI | Case-based discussion |
| 10 | Capstone | Plan and pitch an AI project |

## Repository layout

```
├── index.qmd, setup.qmd, schedule.qmd, instructors.qmd
├── modules/            lecture notes; each page includes its lab
├── resources/          checklist, project canvas, glossary, datasets, further learning
├── labs/python/        lab source scripts (# %% cells) — edit these
├── labs/r/             the same labs in R
├── notebooks/          Colab notebooks generated from labs/ (Python and R)
├── _labs/, figures/    executed lab outputs included in the website
├── data/               simulated datasets + simulate_data.py
├── tools/              build_labs.py, set_repo.py
└── .github/workflows/  publishes the site to GitHub Pages
```

## Publish on GitHub Pages

1. Create a new, empty, **public** repository on GitHub (e.g. `ai-for-agri-bio`), then from this folder:
   ```bash
   git init -b main
   git add .
   git commit -m "Initial course"
   git remote add origin https://github.com/your-username/ai-for-agri-bio.git
   git push -u origin main
   ```
2. Point the Colab buttons and data links at your repository, then commit and push:
   ```bash
   python tools/set_repo.py your-username/ai-for-agri-bio
   git commit -am "Point links at my repository" && git push
   ```
3. On GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
4. The workflow in `.github/workflows/publish.yml` renders the site with Quarto on every push. After a minute or two, the site is live at `https://your-username.github.io/ai-for-agri-bio/`.

The repository must be **public** for Colab to open the notebooks and for the labs to download the data without signing in.

## Work on the course locally

```bash
pip install -r requirements.txt          # Python packages
Rscript install.R                        # R packages
Rscript -e 'IRkernel::installspec()'     # R kernel for Jupyter
python tools/build_labs.py               # rebuild notebooks and outputs (~4 min)
quarto preview                           # live preview of the site
```

See [instructors.qmd](instructors.qmd) for details on editing labs and running the workshop.

## Data

All course data are simulated (`data/simulate_data.py`, fixed seed) to look like real data, including the problems the labs teach you to find. Labs 7A and 7C download public data at run time (PlantVillage via TensorFlow Datasets; UniProt; AlphaFold DB; ESM-2 from Hugging Face).

## License

Course text and figures: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code: MIT. See [LICENSE.md](LICENSE.md).
