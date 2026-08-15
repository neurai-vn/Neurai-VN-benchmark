# Neurai-VN Benchmark: Standardized Machine Learning Models for Multimodal Digital Phenotyping in Mental Health Classification

[![Paper](https://img.shields.io/badge/Paper-arXiv%202026-red?logo=arxiv)](https://arxiv.org/abs/2607.25232) 
[![Code](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/neurai-vn/Neurai-VN-benchmark)
[![Data](https://img.shields.io/badge/Data-Zenodo-blue?logo=zenodo)](https://zenodo.org/records/18976768)

**Abstract:** Digital phenotyping (DP) using smartphones and wearable devices has shown considerable potential for mental health monitoring. However, progress remains difficult to evaluate due to heterogeneous datasets, inconsistent preprocessing pipelines. In this work, we present a reproducible benchmark built upon the Neurai-VN dataset, a multimodal dataset comprising passive sensing and active assessment across multiple temporal scales, from wearable and smartphone devices, collected from 100 Vietnamese adults over two weeks under free-living conditions. We define four binary classification tasks evaluated using standardized subject-wise cross-validation. Representative linear, tree-based, and neural baseline models are evaluated systematically across predefined feature-group configurations. Mean subject-level F1 scores across five cross-validation folds reached 0.71 for Healthy Control vs. Depression and Healthy Control vs. Clinical, while Healthy Control vs. Anxiety and Depression vs. Anxiety achieved 0.69 and 0.56, respectively. These baseline results provide reproducible baselines for future research on multimodal DP for mental health classification tasks. The code to reproduce the benchmark is available at https://github.com/neurai-vn/Neurai-VN-benchmark.

---

## Installation

To run the experiments requires **Python 3.10**, please make sure you have [conda](https://docs.conda.io/en/latest/miniconda.html) installed. Then, create a new conda environment and install the required dependencies:

```bash
conda create -n neurai python=3.10 -y
conda activate neurai
pip install -r requirements.txt
```

## Experiment

The Neurai-VN dataset can be downloaded (latest version) from [Zenodo](https://zenodo.org/records/18976768). Before running the pipeline, configure the dataset paths in `src/configs/neuraivn.json`. 

Mình sẽ viết lại cho README theo kiểu rõ ràng, gọn, và dễ copy-paste hơn:


To run the experiments, use the following command:

```bash
python -m scripts.run_data \
    --db neuraivn \
    --mode benchmarkML-baseline \
    --pipeline_type kfold \
    --task <TASK> \
    --group_feature <GROUP_FEATURE>
```

where `<TASK>` specifies the target classification task and can be one of:

* `hc-dep`
* `hc-anx`
* `dep-anx`
* `hc-clinical`

`<GROUP_FEATURE>` specifies the feature group used in the experiment and can be one of:

* `Wm`
* `Ws`
* `Sd`
* `P`
* `Wm+Ws`
* `P+Sd`
* `Wm+P`
* `Wm+Sd`
* `Ws+P`
* `Ws+Sd`
* `Wm+Ws+Sd`
* `Wm+P+Sd`
* `Ws+P+Sd`
* `Wm+Ws+P`
* `Wm+Ws+P+Sd`


---

## Contact
For technical inquiries, please contact: [24cuong.pq@vinuni.edu.vn](mailto:24cuong.pq@vinuni.edu.vn)

## Citation
- If you use this resource, please cite our paper:
```bibtex
@misc{pham2026neuraivnbenchmarkstandardizedmachine,
      title={Neurai-VN Benchmark: Standardized Machine Learning Models for Multimodal Digital Phenotyping in Mental Health Classification}, 
      author={Quoc-Cuong Pham and Hoang-Thuy-Duong Vu and Thi-Thanh-Huong Ha and Huy-Hieu Pham},
      year={2026},
      eprint={2607.25232},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.25232}, 
}
```
