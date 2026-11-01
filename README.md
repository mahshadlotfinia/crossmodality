# Small labeled samples guide the selection of open vision-language models for medical image analysis

This is the official repository of the paper **Small labeled samples guide the selection of open vision-language models for medical image analysis**.

## Overview

The code asks 38 open-weight vision-language models yes-or-no questions about images of 32 findings in 8 imaging modalities: chest radiography, mammography, funduscopy, dermoscopy, histopathology, CT, MRI, and ultrasound. Each finding has the same question for images with and without it. Image use is measured as the within-finding AUROC of the probability of Yes, normalized over the Yes and No tokens among the 20 most probable initial tokens. The code builds the question sets from public datasets, sends the questions to the models through an OpenAI-compatible endpoint, fits linear heads on DINOv3-L features as a reference, and estimates performance with a cluster bootstrap. A cluster is the dataset's patient group, i.e., a patient, a lesion, a study, a slide, a volume, or an image or tile where a dataset has no patient identifier. The code also simulates the selection of a model from a small labeled sample, relates public benchmark accuracy to image use, and compares models across size series, generations, and medical post-training.

## Installation

```bash
git clone https://github.com/mahshadlotfinia/crossmodality.git
cd crossmodality
```

Install NumPy, SciPy, pandas, scikit-learn, statsmodels, PyYAML, tqdm, the OpenAI Python client, httpx, Pillow, h5py, SimpleITK, pydicom, pylidc, pyarrow, huggingface_hub, PyTorch, and Transformers.

## Configuration

Every path, dataset, model, and setting is in `config/config.yaml`. Replace every `/PATH` with the locations of the datasets and of the output folder. Set `llm_api_base_url` and `llm_api_key` to the address and the key of an OpenAI-compatible server that serves the models, and set `hf_token` to a Hugging Face access token for the gated repositories, or set the environment variables named by `llm_api_base_url_env`, `llm_api_key_env`, and `hf_token_env`. The repetition of the analyses without marked findings reads the table named by `training_data_csv`, which lists the evaluation datasets named in the documented training data of each model.

For LLaVA-Med-7B, `served_name` identifies a local conversion of the Microsoft release to the Hugging Face Transformers format. Its `hf_id` identifies the public conversion by chaoyinshe used to read the parameter count, and `date_from` identifies the original Microsoft repository used to date the release.

The training-data evidence table is a required external input and is not included in this repository. Prepare it from the model cards and reports with the columns `model`, `source`, `quote`, `datasets`, and `modalities`. Include at least one source and its exact quotation for every model in the configured panel. Use the model, dataset, and modality identifiers from the configuration; separate multiple datasets or modalities with semicolons and leave either field empty when the source names none. Supplementary Table 3 summarizes the study's model-to-dataset and model-to-modality assignments. The table is used for both the analysis without marked findings and the comparison of medical-model gains in named and other modalities.

The VinDr adapters read prepared images and labels. For VinDr-CXR, `labels_csv` contains `image_id`, the official `split`, and one binary column per finding. Test labels follow the released consensus; a training label is 1 when at least two distinct readers marked that finding and 0 otherwise. The development adapter excludes any image marked by even one reader from the negatives. Prepare JPEGs under the configured image directory with the layout `<split>/<image_id>.jpg` and retain the original DICOM dimensions for transforming annotations. For VinDr-Mammo, use the breast crops and preprocessing supplied by [Mammo-CLIP](https://github.com/batmanlab/Mammo-CLIP#data-download): its VinDr script converts DICOMs to 8-bit PNGs and resizes breast crops to 912 × 1,520 pixels. Place them under `<study_id>/<image_id>.png` in the configured image directory. The study renderer applies the padding and input size specified in the manuscript.

## Analyses and modules

Each entry point takes the path of `config/config.yaml`. The inference entry points also take the name of a model, and `main_preprocess_download` takes the name of a dataset.

- **Question sets.** `main_preprocess_download` in `data_loader/preprocess_downloads.py` converts a downloaded dataset into the stored images and volumes. `main_build_partitions`, `main_build_question_sets`, `main_build_manifests`, and `main_build_external_sets` in `data_loader/build_question_sets.py` split each dataset by its groups, draw the label-balanced questions of the main and external sets, and write their manifests. `main_build_benchmarks` in `data_loader/benchmarks.py` builds the benchmark questions, and `main_render_inputs` in `data_loader/render_inputs.py` writes the model inputs.
- **Inference.** `main_run`, `main_run_external`, and `main_run_robustness` in `Inference/inference_runner.py` send to one model the questions of the main sets, those of the external sets and the benchmarks, and those of the second phrasing, the lower resolution, and the repeated requests. `main_reparse_records` in `Inference/reparse_records.py` parses stored answers again with the current parser, and `main_model_facts` in `Inference/model_facts.py` counts the parameters and dates the release of every model.
- **Reference.** `main_dinov3_cache` in `encoders/dinov3.py` extracts the DINOv3-L features, and `main_dinov3_heads` in `encoders/heads.py` fits and scores the linear head of each finding.
- **Image use and comparisons.** `main_compute_metrics` in `analysis/compute_metrics.py` computes the within-finding AUROC, the balanced accuracy, and the answer counts on the joint bootstrap draw. `main_contrasts` and `main_scaling_mixedlm` in `analysis/contrasts.py` compute the differences along the size series, the generation and medical pairs, their comparison, and the mixed model. `main_selection` in `analysis/selection.py` simulates the selection from labeled samples and the fixed choices. `main_benchmarks_analysis` in `analysis/benchmarks.py`, `main_externals` in `analysis/externals.py`, and `main_robustness` in `analysis/robustness.py` analyze the benchmarks, the external sets, and the robustness checks. `main_contamination` and `main_scaling_mixedlm_unmarked` in `analysis/contamination.py` repeat the analyses without the findings whose dataset is named in the documented training data of a model, and compute the named-modality differences of the medical pairs. `main_reported_statistics` in `analysis/reported_statistics.py` computes the post hoc differences, the changes per doubling of the parameter count, the regret of the post hoc choice of the model with the highest mean macro AUROC over the other modalities, and the counts, ranges, leaders, and shares reported in the paper.

Every entry point writes its results atomically, resumes an interrupted run, and skips the units it has finished.

## Citation

If you use this repository, please cite our paper:

```bibtex
@article{lotfinia2026small,
  title  = {Small labeled samples guide the selection of open vision-language models for medical image analysis},
  author = {Lotfinia, Mahshad and Maier, Andreas and Tayebi Arasteh, Soroosh},
  year   = {2026}
}
```

## License

MIT License. See `LICENSE` for details.
