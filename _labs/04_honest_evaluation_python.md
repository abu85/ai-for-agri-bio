A model is only as good as the test that measures it. In biology, the most
common mistake is testing on data that are **not truly independent** of the
training data. In this lab you will make that mistake on purpose, then fix it.

- **Part A — Field trials:** overfitting, and leakage between plots from the same site
- **Part B — Population genetics:** relatives and population structure inflate genomic prediction
- **Part C — Metagenomics:** batch effects and feature-selection leakage

The key question is always: *what will the model face when it is used?*
A new plot at a known site? A new site? A new year? A new population?

```python
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.model_selection import (KFold, GroupKFold, LeaveOneGroupOut, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, roc_auc_score

REPO = "YOUR-GITHUB-USERNAME/ai-for-agri-bio"


def data_path(name):
    """Use the local data folder if it exists, otherwise download from GitHub."""
    for folder in ["data", "../data", "../../data"]:
        if os.path.exists(os.path.join(folder, name)):
            return os.path.join(folder, name)
    return f"https://raw.githubusercontent.com/{REPO}/main/data/{name}"


plt.rcParams["figure.figsize"] = (8, 4.5)
SEED = 42


def rmse(true, pred):
    return np.sqrt(mean_squared_error(true, pred))
```

### Part A — Field trials

#### Overfitting

A decision tree can grow until it memorises every training plot. Watch the
training error fall while the test error stops improving and gets worse.

```python
trials = pd.read_csv(data_path("crop_trials.csv"))
trials.loc[trials["yield_t_ha"] > 20, "yield_t_ha"] = np.nan
trials.loc[trials["soil_ph"] < 3, "soil_ph"] = np.nan
trials = trials.dropna(subset=["yield_t_ha"]).reset_index(drop=True)

features = ["n_fert_kg_ha", "soil_p_mg_kg", "soil_k_mg_kg", "soil_ph", "organic_matter_pct",
            "rainfall_mm", "mean_temp_c", "irrigated", "variety"]
X = pd.get_dummies(trials[features], columns=["variety"], dtype=int)
X = X.fillna(X.median())
y = trials["yield_t_ha"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)
depths = range(1, 21)
train_err, test_err = [], []
for d in depths:
    tree = DecisionTreeRegressor(max_depth=d, random_state=SEED).fit(X_train, y_train)
    train_err.append(rmse(y_train, tree.predict(X_train)))
    test_err.append(rmse(y_test, tree.predict(X_test)))

plt.plot(depths, train_err, marker="o", label="training data")
plt.plot(depths, test_err, marker="o", label="test data")
plt.xlabel("tree depth (model complexity)")
plt.ylabel("RMSE (t/ha)")
plt.title("Overfitting: deeper trees memorise the training data")
plt.legend()
plt.show()
```

![](/figures/labs/04_honest_evaluation_python_01.png){.lab-figure}

#### Leakage between plots from the same site

A random split puts plots from the same site in both training and test sets.
The model learns each site's typical yield, and the test looks easy. But if
you want to predict yield at a **new site**, that knowledge doesn't exist.

We compare three cross-validation schemes:

- **Random 5-fold:** plots shuffled at random (the default in most tutorials)
- **Leave-one-year-out:** train on three seasons, predict the fourth
- **Leave-one-site-out:** train on seven sites, predict the eighth

```python
forest = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
schemes = {
    "random 5-fold": dict(cv=KFold(5, shuffle=True, random_state=SEED)),
    "leave-one-year-out": dict(cv=LeaveOneGroupOut(), groups=trials["year"]),
    "leave-one-site-out": dict(cv=LeaveOneGroupOut(), groups=trials["site"]),
}
cv_rmse = {}
for name, kw in schemes.items():
    pred = cross_val_predict(forest, X, y, n_jobs=-1, **kw)
    cv_rmse[name] = rmse(y, pred)
    print(f"{name:20s} RMSE = {cv_rmse[name]:.2f} t/ha")
```

```{.text .lab-output}
random 5-fold        RMSE = 0.98 t/ha
```

```{.text .lab-output}
leave-one-year-out   RMSE = 1.10 t/ha
```

```{.text .lab-output}
leave-one-site-out   RMSE = 1.62 t/ha
```

```python
plt.figure(figsize=(7, 3))
plt.barh(list(cv_rmse), list(cv_rmse.values()), color=["tab:gray", "tab:blue", "tab:red"])
plt.xlabel("cross-validated RMSE (t/ha), lower is better")
plt.title("Same model, same data, three answers")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

![](/figures/labs/04_honest_evaluation_python_02.png){.lab-figure}

Same model, same data — but the error at a new site is much larger than random
cross-validation suggests. **Choose the split that matches how the model will
be used**, and report that number.

**Try it:** Remove `rainfall_mm` and `mean_temp_c` from the features. How does
each CV scheme change? What does that tell you about what the model learned?

### Part B — Population genetics: relatives and population structure

Our plants come from 36 families of full siblings in three landraces. Siblings
share half their DNA. With a random split, most test plants have siblings in
the training set, so predicting them is easier than predicting a new family.
And because the landraces differ in both allele frequencies *and* height, a
model can score well just by recognising the landrace.

```python
geno = pd.read_csv(data_path("genotypes.csv"), index_col="sample_id")
samples = pd.read_csv(data_path("genotype_samples.csv"), index_col="sample_id")
geno = geno.loc[geno.isna().mean(axis=1) <= 0.1, geno.isna().mean(axis=0) <= 0.1]
p = geno.mean() / 2
geno = geno.loc[:, np.minimum(p, 1 - p) >= 0.05]
geno = geno.fillna(geno.mean())
samples = samples.loc[geno.index]
height = samples["plant_height_cm"]

ridge = RidgeCV(alphas=np.logspace(0, 5, 30))
gp_schemes = {
    "random 5-fold": dict(cv=KFold(5, shuffle=True, random_state=SEED)),
    "new families (grouped 5-fold)": dict(cv=GroupKFold(5), groups=samples["family"]),
    "new landrace (leave-one-out)": dict(cv=LeaveOneGroupOut(), groups=samples["population"]),
}
gp_pred = {}
for name, kw in gp_schemes.items():
    gp_pred[name] = cross_val_predict(ridge, geno, height, **kw)
```

We report predictive ability two ways: across all plants, and **within each
landrace** (the average correlation inside each population). Within-landrace
accuracy is what matters when a breeder selects inside one breeding population.

```python
rows = []
for name, pred in gp_pred.items():
    overall = np.corrcoef(height, pred)[0, 1]
    within = np.mean([np.corrcoef(height[m], pred[m])[0, 1]
                      for m in [samples["population"] == pop for pop in samples["population"].unique()]])
    rows.append(dict(scheme=name, r_all_plants=round(overall, 2), r_within_landrace=round(within, 2)))
pd.DataFrame(rows)
```

```{=html}
<div class="lab-table"><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>scheme</th>
      <th>r_all_plants</th>
      <th>r_within_landrace</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>random 5-fold</td>
      <td>0.63</td>
      <td>0.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>new families (grouped 5-fold)</td>
      <td>0.53</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>new landrace (leave-one-out)</td>
      <td>-0.35</td>
      <td>0.21</td>
    </tr>
  </tbody>
</table>
</div></div>
```

Three lessons:

1. Across all plants, random CV looks best, partly because differences
   *between* landraces are easy to predict once every landrace is in the
   training data.
2. Within landraces, accuracy is much lower — and it drops sharply for new
   families, because random CV let the model "cheat" through siblings.
3. For a landrace the model has never seen, the across-plant correlation can
   collapse or even turn negative: the model cannot know that landrace's
   average height, which here depends partly on its environment of origin.

Published genomic prediction accuracies are only comparable if you know which
scheme was used.

**Try it:** Use `flowering_days` as the trait. Do the same patterns appear?

### Part C — Metagenomics: batch effects and feature-selection leakage

#### Batch confounding

Samples were sequenced in two batches — and, by accident of lab logistics,
most diseased samples went into batch 2.

```python
counts = pd.read_csv(data_path("microbiome_counts.csv"), index_col="sample_id")
meta = pd.read_csv(data_path("microbiome_metadata.csv"), index_col="sample_id")
taxonomy = pd.read_csv(data_path("microbiome_taxonomy.csv"), index_col="taxon_id")
counts = counts.loc[:, (counts > 0).mean() >= 0.1]
logx = np.log(counts + 0.5)
clr = logx.sub(logx.mean(axis=1), axis=0)
status = (meta["soil_status"] == "diseased").astype(int)

pd.crosstab(meta["soil_status"], meta["seq_batch"])
```

```{=html}
<div class="lab-table"><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>seq_batch</th>
      <th>batch1</th>
      <th>batch2</th>
    </tr>
    <tr>
      <th>soil_status</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>diseased</th>
      <td>28</td>
      <td>44</td>
    </tr>
    <tr>
      <th>healthy</th>
      <td>52</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div></div>
```

*Methylobacterium* and *Sphingomonas* are typical kit contaminants: they come
from the extraction reagents, not the soil, and here they are higher in batch 2.
Watch what happens if we predict disease from **contaminants only**:

```python
contaminants = [t for t in clr.columns if taxonomy.loc[t, "genus"] in ("Methylobacterium", "Sphingomonas")]
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)


def cv_auc(features, labels, cv=cv, groups=None):
    prob = cross_val_predict(rf, features, labels, cv=cv, groups=groups, method="predict_proba")[:, 1]
    return roc_auc_score(labels, prob)


print(f"all taxa:            AUC = {cv_auc(clr, status):.2f}")
print(f"contaminants only:   AUC = {cv_auc(clr[contaminants], status):.2f}")
print(f"contaminants removed: AUC = {cv_auc(clr.drop(columns=contaminants), status):.2f}")
```

```{.text .lab-output}
all taxa:            AUC = 0.95
```

```{.text .lab-output}
contaminants only:   AUC = 0.68
```

```{.text .lab-output}
contaminants removed: AUC = 0.99
```

Contaminants alone predict "disease" well above chance (AUC 0.5) — not because
they have anything to do with plant health, but because they mark the batch,
and the batch is linked to disease status. **A model will happily learn any
shortcut you leave in the data.** The fix starts at the design stage: randomise
samples across batches, sequence negative controls, and record batch in the
metadata.

A tougher, more honest test also asks whether the model works in a **new field**:

```python
no_contam = clr.drop(columns=contaminants)
print(f"random 5-fold:        AUC = {cv_auc(no_contam, status):.2f}")
print(f"leave-one-field-out:  AUC = {cv_auc(no_contam, status, cv=LeaveOneGroupOut(), groups=meta['field']):.2f}")
```

```{.text .lab-output}
random 5-fold:        AUC = 0.99
```

```{.text .lab-output}
leave-one-field-out:  AUC = 0.96
```

#### Feature-selection leakage

Omics data have many more features than samples. A tempting shortcut is to pick
the "most significant" taxa using **all** samples, then cross-validate a model
on those taxa. To see why this is wrong, we **shuffle the labels**, so there is
no real signal at all, and compare two procedures:

- **Wrong:** select the top 10 taxa on all data, then cross-validate
- **Right:** select the top 10 taxa *inside* each training fold (a pipeline)

```python
rng = np.random.default_rng(SEED)
logreg = LogisticRegression(max_iter=1000)
wrong_auc, right_auc = [], []
for _ in range(20):  # repeat with 20 different label shuffles
    fake = pd.Series(rng.permutation(status.values), index=status.index)
    top10 = SelectKBest(f_classif, k=10).fit(clr, fake).get_support()
    wrong = cross_val_predict(logreg, clr.loc[:, top10], fake, cv=cv, method="predict_proba")[:, 1]
    right = cross_val_predict(make_pipeline(SelectKBest(f_classif, k=10), logreg), clr, fake,
                              cv=cv, method="predict_proba")[:, 1]
    wrong_auc.append(roc_auc_score(fake, wrong))
    right_auc.append(roc_auc_score(fake, right))

print(f"random labels, selection outside CV: mean AUC = {np.mean(wrong_auc):.2f}  <- looks like signal")
print(f"random labels, selection inside CV:  mean AUC = {np.mean(right_auc):.2f}  <- the truth: chance")
```

```{.text .lab-output}
random labels, selection outside CV: mean AUC = 0.68  <- looks like signal
random labels, selection inside CV:  mean AUC = 0.55  <- the truth: chance
```

```python
plt.figure(figsize=(7, 3))
plt.boxplot([wrong_auc, right_auc], vert=False)
plt.yticks([1, 2], ["selection outside CV\n(wrong)", "selection inside CV\n(right)"])
plt.axvline(0.5, color="red", ls="--", label="chance")
plt.xlabel("cross-validated AUC on shuffled labels (20 repeats)")
plt.legend()
plt.tight_layout()
plt.show()
```

![](/figures/labs/04_honest_evaluation_python_03.png){.lab-figure}

With no real signal, the wrong procedure still reports a model that "works".
**Every step that looks at the labels — feature selection, tuning, even some
normalisations — must happen inside the training folds.**

### Checklist for your own projects

- Are related samples (same plant, plot, site, family, patient, batch) kept on
  the same side of the split?
- Does the test scenario match the real use (new site, new year, new population)?
- Is there a simple baseline to beat?
- Were all label-dependent steps done inside cross-validation?
- Could a technical variable (batch, date, operator) be the real "signal"?
