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
Data preparation
```bash
python3 data_valid.py 
```
Initialization
```bash
python3 initializationS.py --
```



## References
```
TBD
```

Bibtex entry ready to be cited
```
TBD
```
