# Landsat / Sentinel Urban Classification Project

This repository contains the code and artifacts used for the urban land-cover workflow described in the paper:

1. Google Earth Engine data extraction for Landsat imagery.
2. Spectral feature generation with RGB, NDVI, and NDBI.
3. Random Forest experiments for classification and prediction.
4. Cloud-based notebook workflows using Google Drive / Colab.
5. Post-processing, visualization, and web presentation of the results.

## Current Structure

```text


archive/
  legacy_artifacts/

data/
  vector/
    BENGALURU.geojson
  rasters/
    Bengaluru_builtup_mask.tif
    Bengaluru_labeled_mask.tif
    Bengaluru_labeled_mask_2022_FINAL.tif
    Bengaluru_labeled_mask_2023_FINAL.tif
    Bengaluru_labeled_mask_2024_FINAL.tif
    Bengaluru_mask_updated.tif
    Sentinel_RGBNIR_2024.tif

docs/
  figures/
    image.png
  paper/
    IEEE_Conference.pdf
notebooks/
  cloud_workflows/
    generalized_ca_markov_prediction.ipynb
    markup_classifier_cloud_workflow.ipynb
    predict.ipynb

scripts/
  preprocessing/
  classification/
    logistic_regression.py
    random_forest/
  visualization/


```

## Notes

- The cloud notebooks are kept in `notebooks/cloud_workflows/` because they were used to run the Drive-backed experiments.
- Random Forest scripts now have descriptive names under `scripts/classification/random_forest/`.
- Preprocessing and label-generation code lives under `scripts/preprocessing/`.
- Visualization scripts and animation scenes live under `scripts/visualization/` and `animations/`.
- The normalization diagram artifact was moved to `archive/legacy_artifacts/` so the repository root stays focused on the paper workflow.
- The paper PDF was not present in the workspace snapshot, so `docs/paper/` is ready for it when you add the file.

