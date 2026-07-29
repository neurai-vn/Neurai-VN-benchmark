# Neurai-VN Benchmark

Benchmarking machine learning models on the Neurai-VN dataset.

This repository provides the preprocessing pipeline, handcrafted feature extraction, subject-level data splitting, and baseline machine learning benchmarks for the Neurai-VN dataset.

## Citation

If you use this repository in your research, please cite our paper:
```bibtex
@article{,
  title   = {Neurai-VN Benchmark: Standardized Machine Learning Models for Multimodal Digital Phenotyping in Mental Health Classification},
  author  = {Quoc-Cuong Pham, Hoang-Thuy-Duong Vu, Thi-Thanh-Huong Ha, Huy-Hieu Pham},
  journal = {arXiv preprint arXiv:2607.25232},
  year    = {2026}
}
```

Preprint: https://arxiv.org/pdf/2607.25232

## Requirements
- Python 3.10
- Install the required packages: 

```bash
pip install -r requirements.txt
```

## Configuration

Before running the pipeline, configure the dataset paths in:
```text
src/configs/neuraivn.json
```

| Key | Description |
|------|-------------|
| `DIR_RAW` | Path to the Neurai-VN dataset downloaded from the original source. |
| `DIR_PROCESSED` | Directory for processed outputs (e.g., `sensor_raw/`, `feature_original/`, `splits/`). |
| `DIR_RESULTS_ML_BASELINE` | Directory for baseline ML benchmark results. |


| Key | Description |
|------|-------------|
| `DIR_RAW` | Path to the Neurai-VN dataset downloaded from the original source. |
| `DIR_PROCESSED` | Directory for processed outputs (e.g., `sensor_raw/`, `feature_original/`, `splits/`). |
| `DIR_RESULTS_ML_BASELINE` | Directory for baseline ML benchmark results. |


## 1. Data Status

Check data availability and labels.

```bash
# Check one raw file of a subject
python -m scripts.run_data --db neuraivn --mode checkfile --key P0001-activeZoneMinutes.csv

# Check all raw files of a subject
python -m scripts.run_data --db neuraivn --mode checkfile --key P0001-all

# Check labels
python -m scripts.run_data --db neuraivn --mode label
```

---

## 2. Data Splitting

Generate subject-level data splits.

**Output**

```text
assets/neuraivn/splits/
└── seed_42.json
```

**Run**

```bash
python -m scripts.run_data --db neuraivn --mode split --key all
```

---

## 3. Sensor Extraction

Extract raw sensor data for each subject.

**Output**

```text
assets/neuraivn/sensor_raw/
├── P0001.pkl
├── P0002.pkl
└── ...
```

**Run**

```bash
# Single subject
python -m scripts.run_data --db neuraivn --mode sensor --key P0001

# All subjects
python -m scripts.run_data --db neuraivn --mode sensor --key all --use_ray True
```

---

## 4. Original Feature Extraction

Extract handcrafted features for baseline ML models.

**Output**

```text
assets/neuraivn/feature_original/
├── P0001.pkl
├── P0002.pkl
└── ...
```

**Run**

```bash
# Single subject
python -m scripts.run_data --db neuraivn --mode feature-original --key P0001

# All subjects
python -m scripts.run_data --db neuraivn --mode feature-original --key all --use_ray True
```



## 7. Benchmark ML
### Description
Run baseline ML models using **5-fold cross-validation**. Results are saved to:
```text
results/benchmarkML/baseline_<database>/
├── <group_feature>-<task>_5foldcv.csv
├── ...
```

For example:

```text
results/benchmarkML/baseline_neuraivn/
├── Wm-hc-clinical_5foldcv.csv
├── Wm+Ws-hc-dep_5foldcv.csv
├── ...
```

### Tasks

- `hc-dep`
- `hc-anx`
- `dep-anx`
- `hc-clinical`

### Feature Groups

- `Wm`
- `Ws`
- `Sd`
- `P`

**2-group combinations**

- `Wm+Ws`
- `P+Sd`
- `Wm+P`
- `Wm+Sd`
- `Ws+P`
- `Ws+Sd`

**3-group combinations**

- `Wm+Ws+Sd`
- `Wm+P+Sd`
- `Ws+P+Sd`

**4-group combination**

- `Wm+Ws+P+Sd`

### Usage

```bash
python -m scripts.run_data \
    --db neuraivn \
    --mode benchmarkML-baseline \
    --pipeline_type kfold \
    --task hc-clinical \
    --group_feature Wm
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--db` | Dataset name (e.g., `neuraivn`) |
| `--mode` | Benchmark mode (`benchmarkML-baseline`) |
| `--pipeline_type` | Evaluation pipeline (`kfold`) |
| `--task` | Classification task |
| `--group_feature` | Feature group to evaluate |
