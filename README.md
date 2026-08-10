# Multimodal Late-Fusion Classifier with Explainable AI for Alzheimer's Disease Staging

**Kishen Patel** · BSc Computer Science, University of Bath (2024)

A late-fusion multimodal classifier that stages patients as cognitively normal (CN), mildly cognitively impaired (MCI), or diagnosed with Alzheimer's disease (AD), combining clinical, genetic, and biomarker data from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/). Three modality-specific models are trained independently and fused through a meta-model, with explainable AI (XAI) applied throughout to keep predictions interpretable.

Originally developed as an undergraduate dissertation (BSc Computer Science, University of Bath, 83/100), since revised into a journal manuscript.

📄 Paper: *link to arXiv preprint / journal submission, once available*
🔗 Portfolio write-up: *link to your website project page*

---

## Overview

Alzheimer's disease is difficult to stage accurately from any single data source. This project combines three modalities - each with its own model architecture - and fuses their predictions:

| Modality | Model | What it captures |
|---|---|---|
| **Clinical** | Feedforward Neural Network (FNN) | Cognitive scores, neurological exam results, brain-volume measures (ADNIMERGE) |
| **Genetic** | Gradient Boosting Machine (GBM) | Variant counts across a curated list of AD-associated genes, filtered from whole-genome VCF files |
| **Biomarker** | Recurrent Neural Network (LSTM) | Four blood biomarkers tracked across five longitudinal timepoints |

The three models' class-probability outputs are concatenated into a single feature vector and passed to a meta-model (Logistic Regression / Random Forest) for the final diagnosis.

**Best result:** 98.0% accuracy on a held-out test set of 51 patients (419-patient subset, Logistic Regression, plain concatenation) - see the paper for the full result table, confidence intervals, and discussion of the small-test-set caveat.

## Key features

- **Targeted genetic preprocessing.** Rather than processing ~3 million raw SNPs per patient, variants are filtered against a curated list of AD-associated genes (via AlzForum's AlzPedia), reducing the input to a compact, interpretable set of per-gene counts.
- **Multimodal late fusion.** Each modality is modelled independently before fusion, letting each architecture specialise rather than forcing one model to learn across three very different data types at once.
- **Explainability throughout.** SHAP, LIME, partial dependence plots, mutual information ranking, and gradient-based saliency maps are applied per modality - not bolted on afterward - so every prediction can be traced back to the features that drove it.

## Repository structure

```
.
├── Multimodal Dataset Preprocessing.ipynb   # Run this first - builds all datasets
├── Clinical Classification.ipynb            # FNN + Yellowbrick/LIME/SHAP
├── Genetic Classification.ipynb             # GBM + Yellowbrick/PDPs/Shapash
├── Biomarker Classification.ipynb           # RNN (LSTM) + Yellowbrick/GradientTape
├── Multimodal Classification.ipynb          # Fusion: LR / RF meta-models
└── README.md
```

## Requirements

- Python 3.11.5+
- `tensorflow` 2.16.1
- `keras` 3.2.1
- `scikit-learn` 1.3.2
- `imbalanced-learn` 0.12.0
- `scikit-image` 0.20.0
- `pandas` 2.0.3
- `numpy` 1.24.3
- `seaborn` 0.12.2
- `matplotlib`
- `lime`
- `shap`
- `shapash`
- `yellowbrick`

Install with:

```bash
pip install tensorflow==2.16.1 keras==3.2.1 scikit-learn==1.3.2 imbalanced-learn==0.12.0 \
            scikit-image==0.20.0 pandas==2.0.3 numpy==1.24.3 seaborn==0.12.2 \
            matplotlib lime shap shapash yellowbrick
```

A `requirements.txt` is included for convenience - `pip install -r requirements.txt`.

## Data access

This project uses data from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/). **Data is not included in this repository** and must be requested directly from ADNI by qualified researchers via the [Image and Data Archive (IDA)](https://ida.loni.usc.edu/).

Files expected locally (not tracked in this repo - see `.gitignore`):

| File / folder | Contents |
|---|---|
| `LOCLAB_22Apr2024.csv` (split by timepoint: BL, M12, M24, M36, M48) | Biomarker data |
| `GeneticDataVCFs/` | Whole-genome VCF files |
| `ADNIMERGE.CSV` | Clinical data |

Update the file paths at the top of each notebook to point to your local copies before running.

## Usage

1. **Run `Multimodal Dataset Preprocessing.ipynb` first.** This merges and cleans all three modalities, imputes missing values (ExtraTreesRegressor + IterativeImputer), applies class-balancing via random oversampling, and pickles the processed datasets for the other notebooks to consume.

   - By default, only patients with both clinical and biomarker data are retained (419-patient subset).
   - To use the full 818-patient cohort instead, remove `'CTRED'` from this line in the "Combining All Datasets" section:
     ```python
     merged_dataset.dropna(subset=['PTEDUCAT', 'CTRED'], inplace=True)
     ```

2. **Run the three unimodal notebooks** (`Clinical Classification.ipynb`, `Genetic Classification.ipynb`, `Biomarker Classification.ipynb`) in any order. Each trains its respective model and generates its XAI outputs independently.

3. **Run `Multimodal Classification.ipynb`** to fuse the three modalities' predictions and train the meta-models.

## Results summary

| Model | Accuracy | F1 |
|---|---|---|
| Clinical (unimodal) | 92.0% | 92.0% |
| Genetic (unimodal) | 87.3% | 87.0% |
| Biomarker (unimodal) | 61.5% | 60.0% |
| **Fused (LR, plain concatenation, 419 subset)** | **98.0%** | **98.0%** |

*(95% Wilson CI on the fused result: 89.7–99.7%, given the 51-patient test set - see the paper for full discussion of this limitation.)*

## Limitations

- Test set for the fused model is small (51 patients); metrics should be read as indicative for this cohort rather than a settled, generalisable result.
- Missing values were imputed before the train/test split (feature-only, not label-informed) - see the paper for a full discussion of this design choice.
- Trained and evaluated on ADNI1 only; no external cohort validation yet.
- Fusion weights for the "weighted concatenation" variant were set by hand rather than learned.

Full discussion, confidence intervals, and comparison against published multimodal AD studies are in the accompanying paper.

## Citation

If you use this code or build on this work, please cite:

```bibtex
@misc{patel2024multimodal,
  author       = {Patel, Kishen},
  title        = {A Multimodal Late-Fusion Classifier with Explainable AI for Staging Alzheimer's Disease from Clinical, Genetic, and Biomarker Data},
  year         = {2024},
  howpublished = {\url{PASTE_ARXIV_OR_REPO_LINK_HERE}}
}
```

## Acknowledgements

Data collection and sharing for this project was funded by the Alzheimer's Disease Neuroimaging Initiative (ADNI). ADNI investigators contributed to the design and implementation of ADNI and provided data but did not participate in the analysis or writing of this project.

The classifier evaluation loop was developed with guidance and adapted from [rsinghlab/MADDi](https://github.com/rsinghlab/MADDi).

## Contact

Kishen Patel - www.kishenxpatel.com
