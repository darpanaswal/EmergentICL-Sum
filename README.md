# Exploring Cross-Lingual Transfer Learning Techniques for Abstractive Text Summarization

This repository contains the code and resources for the paper titled "Exploring Cross Lingual Transfer Learning Techniques for Abstractive Text Summarization". This research investigates the emergent properties of in-context learning in large language models for multilingual and cross-lingual summarization tasks.

## 📝 Paper Abstract

Cross-lingual transfer learning enhances abstractive text summarization across languages. This study examines whether in-context learning (ICL) is an inherent property of transformers or an emergent feature of advanced large language models (LLMs). We compare Google mT5-xl, Facebook mBART50, and GPT-40-mini in zero-shot, one-shot, and two-shot settings for multilingual and cross-lingual summarization. Fine-tuning improves task-specific performance, with mBART50-English excelling. While transfer learning benefits same-language summarization, cross-lingual tasks suffer due to instruction-following limitations in older models. Unlike earlier transformers, GPT-40-mini exhibits strong zero-shot performance and benefits from one-shot prompting, though additional examples provide minimal gains. These findings suggest that ICL is an emergent property of modern LLMs, informing future advancements in cross-lingual NLP.

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* Python 3.8+
* PyTorch
* Transformers
* Datasets
* PEFT (Parameter-Efficient Fine-Tuning)
* bitsandbytes
* rouge-score
* bert-score

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/cross-lingual-summarization.git](https://github.com/your-username/cross-lingual-summarization.git)
    cd cross-lingual-summarization
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file is not provided, but one should be created with the libraries mentioned above.)*

3.  **Hugging Face Authentication:**
    You will need a Hugging Face token to download the models.
    ```python
    from huggingface_hub import login
    login()
    ```

## 📂 Repository Structure

````

.
├── datasets/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── train\_fr.csv
│   ├── val\_fr.csv
│   ├── test\_fr.csv
│   ├── train\_cross.csv
│   ├── val\_cross.csv
│   └── test\_cross.csv
├── download\_datasets.py
├── download\_model.py
├── finetune.py
├── load\_finetuned.py
├── main.py
├── metrics.py
└── README.md

````

## 📊 Experiments

This project evaluates three models (mT5-xl, mBART50, and GPT-40-mini) on English-to-English, French-to-French, and French-to-English summarization tasks.

### 1. Download the Datasets

First, download and preprocess the WikiLingua dataset by running:

```bash
python download_datasets.py
````

This script will create the necessary `.csv` files in the `datasets/` directory.

### 2\. Fine-tuning the Models

To fine-tune the `mT5` and `mBART50` models, use the `finetune.py` script.

**Usage:**

```bash
python finetune.py --model [MODEL_NAME] --finetune_type [FINETUNE_TYPE]
```

**Arguments:**

  * `--model`: The model to fine-tune. Choices: `mT5`, `mBART50`.
  * `--finetune_type`: The type of fine-tuning. Choices: `english`, `multilingual`, `crosslingual`.

**Examples:**

  * Fine-tune mT5 on the English dataset:
    ```bash
    python finetune.py --model mT5 --finetune_type english
    ```
  * Fine-tune mBART50 for cross-lingual summarization:
    ```bash
    python finetune.py --model mBART50 --finetune_type crosslingual
    ```

### 3\. Running Evaluations

After fine-tuning, you can run the zero-shot and few-shot evaluations using the `main.py` script.

**Usage:**

```bash
python main.py --model [MODEL_NAME] --experiment [EXPERIMENT_TYPE] --finetune_type [FINETUNE_TYPE] [--num_examples NUM_EXAMPLES]
```

**Arguments:**

  * `--model`: The model to evaluate. Choices: `mT5`, `mBART50`.
  * `--experiment`: The type of experiment to run. Choices: `zero-shot`, `1-shot`.
  * `--finetune_type`: The fine-tuning of the model to load. Choices: `english`, `multilingual`, `crosslingual`.
  * `--num_examples` (optional): The number of examples from the test set to evaluate on.

**Examples:**

  * Run a zero-shot evaluation on the English fine-tuned mT5 model:
    ```bash
    python main.py --model mT5 --experiment zero-shot --finetune_type english
    ```
  * Run a 1-shot evaluation on the multilingual fine-tuned mBART50 model on 100 examples:
    ```bash
    python main.py --model mBART50 --experiment 1-shot --finetune_type multilingual --num_examples 100
    ```

## 📈 Results

The results of the experiments are saved to `rouge_results.csv`. The paper's findings indicate that while fine-tuning significantly improves the performance of `mT5` and `mBART50`, they do not benefit from in-context learning. In contrast, `GPT-4o-mini` shows strong zero-shot performance and benefits from one-shot prompting, suggesting that in-context learning is an emergent property of modern large language models.

## ✍️ Citation

If you use this code or find the research useful, please cite the paper:

```bibtex
@article{aswal2025crosslingual,
  title={Exploring Cross Lingual Transfer Learning Techniques for Abstractive Text Summarization},
  author={Aswal, Darpan},
  journal={Preprint},
  year={2025}
}
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome\!

## 📞 Contact

Darpan Aswal - darpan.aswal@universite-paris-saclay.fr
```
```
