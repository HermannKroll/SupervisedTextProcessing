# A Library Perspective on Supervised Text Processing in Digital Libraries: An Investigation in the Biomedical Domain

This repository is part of our JCDL2024 submission. You can find the publication under [link will follow]().

Please cite our following paper when working with our repository.
```
@inproceedings{kroll2021toolbox,
  author = {H. Kroll, P. Sackhoff, M. Thang, M. Ksouri and W.-T. Balke},
  booktitle = {2024 ACM/IEEE Joint Conference on Digital Libraries (JCDL)},
  title = {A Library Perspective on Supervised Text Processing in Digital Libraries: An Investigation in the Biomedical Domain},
  year = {2024}
}
```

## Documentation

The implementation is in the [src](/src) folder. 
The following modules are provided:

- [Analysis](/src/narrarelex/analysis) - various analysis scripts
- [Data Generation](/src/narrarelex/data_generation) - utilized datasets in our format
- [Prediction](/src/narrarelex/prediction) - processing-pipelines for the traditional and BERT models
- [Training](/src/narrarelex/training) - implementation of the model training

For the evaluation the following scripts were used:

- [evaluate_bert.py](/src/narrarelex/evaluate_bert.py) pipeline for the BERT language models
- [evaluate_traditional.py](/src/narrarelex/evaluate_traditional.py) pipeline for the traditional classification models
- [evaluate_hs.py](/src/narrarelex/evaluate_hs.py) script to evaluate the hyperparameter search
- [evaluate_text_classification_noise.py](/src/narrarelex/evaluate_text_classification_noise.py) script to evaluate the noise impact of the TC task
- [run_config.py](/src/narrarelex/run_config.py) contains the configuration parameters required by each evaluation script
- [config.py](/src/narrarelex/config.py) contains directory organization

## Project setup

1. The project is implemented in python.
For that, we used Conda to create a new py environment:

```shell
conda create -n stpenv python=3.8
conda activate stpenv
```

2. To reproduce the results of our evaluation, you first need to install the required python libraries. 

```shell
pip -r requirements.txt
```

3. Set the python path to our src root.

```shell
export PYTHONPATH="/home/USER/SupervisedTextProcessing/src/
```

4. We publish our self-curated dataset [Pharmaceutical Technologies](/data/resources/benchmark/PharmaTech.zip) along with the code.
To use it in the evaluation, it has to be decompressed first. 
The script will extract the data into the correct location.

```shell
python src/narrarelex/init_benchmarks.py
```

5. To use the relabeling module correctly, you first need to paste your API keys into the files [HUGGINGFACE_TOKEN](HUGGINGFACE_TOKEN) and [OPENAI_TOKEN](OPENAI_TOKEN).

## Benchmarks

### Task 1: Relation Extraction
> - CDR [GitHub](https://github.com/JHnlp/BioCreative-V-CDR-Corpus)
> - ChemProt [Website](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/)
> - DDI [GitHub](https://github.com/albertrial/SemEval-2013-task-9)

Please note, that our Chemprot variants, called as ChemprotE and ChemprotE, are created at runtime.

### Task 2: Text Classification
> - HallmarksOfCancer [Website](https://autonlp.ai/datasets/hoc-(hallmarks-of-cancer))
> - Ohsumed [Website](https://disi.unitn.it/moschitti/corpora.htm)
> - Long Covid [Website](https://huggingface.co/datasets/llangnickel/long-covid-classification-data)
> - PharmaTech [Repository](/data/resources/benchmark/PharmaTech.zip)
