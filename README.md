# Automatic Coronary Calcium Scoring from Gated Coronary CT Using DL-based FP detection model
CAC Scoring from NCCT Using DL With External Validation

![check](methods_overview.png)
This work introduces an automatic CAC scoring method that uses multi-atlas segmentation for whole heart segmentation (WHS) and a DL model as a supervised classifier for correcting false positives (FP). <br/>


## Descriptions
- The repository provides a DL-based FP (of CAC) model.
- The model is developed by using the [Stanford AIMI COCA dataset](https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa) which is publicly available for research purpose
- We used the multi-atlas segmentation pipeline implemented by the [Biomedical Imaging Group Rotterdam (BIGR)](https://bigr.nl/)
- Our work was externally validated on the [Rotterdam Study](https://pubmed.ncbi.nlm.nih.gov/38324224/)

## Run
Generate patches labeled patches with annotated images 
```bash
python3 patch_prep.py -patch_size 45
```
Split the patch data into non-overlapping 5 folds w.r.t subjects  
```bash
python3 k-fold_prep.py -normalize
```
Evaluate binary classification performance and save the trained models
```bash
python3 fp_classifier_train_subject_fold.py -batch_size 32 -n_epochs 100 -lr 1e-4
```
Compute CAC scores
```bash
python3 coca_internal_eval.py -trained_model 'fp_vgg_trained_model_3.pth'
```
Assess the agreement between computed scores and reference scores
```bash
python3 coca_score_agreement.py
```

## References
```
TBD
```

Bibtex entry ready to be cited
```
TBD
```
