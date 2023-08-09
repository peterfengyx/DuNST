# DuNST
Official script of [ACL 2023 paper: DuNST: Dual Noisy Self Training for Controllable Text Generation](https://aclanthology.org/2023.acl-long.488.pdf)

## Introduction

Self-training (ST) has prospered again in language understanding by augmenting the fine-tuning of big pre-trained models when labeled data is insufficient. However, it remains challenging to incorporate ST into attribute-controllable language generation. Augmented only by self-generated pseudo text, generation models over-exploit the previously learned text space and fail to explore a larger one, suffering from a restricted generalization boundary and limited controllability. In this work, we propose DuNST, a novel ST framework to tackle these problems. DuNST jointly models text generation and classification as a dual process and further perturbs and escapes from the collapsed space by adding two kinds of flexible noise. In this way, our model could construct and utilize both pseudo text generated from given labels and pseudo labels predicted from available unlabeled text, which are gradually refined during the ST phase. 

## Repository
DuNST
├── data
├── corpus
├── codes
├── (unilm)
└── (your evaluation classifier)

## Data
You can download the training data of [IMDb](https://huggingface.co/datasets/imdb), [AGNews](https://huggingface.co/datasets/ag_news) from Huggingface. [Jigsaw](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/) dataet can be found on Kaggle.

We use UniLM1-base-cased for our base model. Please download it from the following [link](https://github.com/microsoft/unilm/tree/master/unilm-v1).

## Training
This code can be ran with single GPU. Script that works on multi GPU is on process. 
Simply run [codes/train.py] to replicate our experimental result.
You are free to play with the hyperparameters and settings in [codes/config.py].

## Evaluation/Inference
[codes/evaluation.py] evaluates the classification performance of trained model (F1) and generalizability of generation (Model PPL).
[codes/generation.py] generates samples of given prompt and evaluates the fluency (Output PPL), classification, and diversity (Dist, Self-BLEU).


## License

This repository is licensed under the [MIT License](LICENSE). 

## Citation

If you find our work useful, please consider citing our ACL paper:

```
@inproceedings{feng-etal-2023-dunst,
    title = "{D}u{NST}: Dual Noisy Self Training for Semi-Supervised Controllable Text Generation",
    author = "Feng, Yuxi  and
      Yi, Xiaoyuan  and
      Wang, Xiting  and
      Lakshmanan, V.S., Laks  and
      Xie, Xing",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.488",
    doi = "10.18653/v1/2023.acl-long.488",
    pages = "8760--8785",
}
```
