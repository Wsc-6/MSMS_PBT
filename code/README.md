# MS/MS-to-PBT screening

Core source code for predicting a biodegradability-based persistence screen, bioconcentration and aquatic toxicity from tandem mass spectra. The persistence endpoint is an operational screening label rather than a regulatory persistence classification.

This repository contains analysis source and configuration. The manuscript's Supplementary Data, Source Data and Supplementary Code 1 provide the separate data and full frozen reproduction archive.

## Contents

- `src/pbt_msms/`: 31 current modules for spectrum processing, models, evaluation, calibration, interpretation and field prioritization.
- `建模分析/`: 41 analysis notebooks covering label aggregation, structure-group splits, model comparisons, independent evaluation and field applications. Notebook source is retained with execution outputs cleared. Several early and training notebooks contain implementations not duplicated in the modules.
- `configs/default.yaml`: input paths, endpoint definitions, split rules and model settings.
- `analysis/external_common_set/`: source, analysis plan and pinned dependencies for the retrospective external comparison reported in Supplementary Table S16.
- `tests/`: existing pipeline and nested-calibration checks.
- `SHA256SUMS.csv` and `verify_core.py`: integrity checks for this source snapshot.

The repository excludes frozen prediction tables, spectra, trained weights, figure inputs and generated plots. The full Supplementary Code 1 archive retains the current figure renderers, their inputs and frozen evaluation records.

## Installation and checks

Use Python 3.10 or later in a dedicated environment:

```sh
python -m pip install -e ".[dev]"
python verify_core.py
python -m unittest discover -s tests -v
pbt-msms --help
```

Deep models need `.[deep]`. Some notebooks additionally need `.[plot,extended]`. DreaMS source and pretrained weights, SIRIUS/CSI:FingerID, CFM-ID and ProteoWizard/msconvert are provisioned separately where used.

## Analysis entrypoints

Notebook numbers follow the analysis sequence: 01–03 aggregate labels and freeze cohorts/spectral inputs; 04–19 implement model training and internal comparisons; 20 and 38 evaluate the independent laboratory cohort; 21–37 process the two field applications; 39–41 assess interpretation, confounding and scoring sensitivity.

Set the project inputs in `configs/default.yaml` and the project/executable paths in the relevant notebooks before executing them. Some notebooks retain original local path defaults. The CLI runs the initial notebook stages, for example:

```sh
pbt-msms prepare --config configs/default.yaml
pbt-msms split --config configs/default.yaml
```

Full training requires the original spectral/property resources, processed arrays, pretrained weights and fitted checkpoints described with the manuscript; these resources are not bundled in this source repository.

For the standalone external comparison, copy the seven-input `inputs` directory from `Code/Revision_20261026/S31_reproduction/` inside Supplementary Code 1 to `analysis/external_common_set/inputs/`. Then use its pinned requirements and run:

```sh
python -m pip install -r analysis/external_common_set/requirements.txt
python analysis/external_common_set/reanalyze_external.py
```

The script checks input hashes and computes descriptive intervals from frozen predictions without fitting or selecting a model. Historical S31 computation identifiers refer to publication Table S16. The separate Figure 4c positive-gain tests and Holm correction are documented in the full Supplementary Code 1 archive.

## Frozen analysis conventions

The corrected internal P calibration and the original frozen deployment maps used for external/field predictions are distinct. Their frozen records are retained in Supplementary Code 1; external results should not be recalculated by substituting the revised internal map. Bangladesh scoring uses `E=(o+a)/2` and `S=(0.70H+0.30E)R`, as described in the SI.
