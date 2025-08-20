Cross-Lingual Summarization with Transfer Learning & In-Context Learning

This repository contains the code and experiments for the paper:
“Exploring Cross-Lingual Transfer Learning Techniques for Abstractive Text Summarization”
by Darpan Aswal, Université Paris-Saclay.

We investigate whether In-Context Learning (ICL) is an inherent property of transformer models or an emergent capability of large language models (LLMs), with a focus on cross-lingual abstractive summarization.

We compare Google mT5-xl, Facebook mBART50, and GPT-4o-mini across zero-shot, one-shot, and two-shot summarization tasks on the WikiLingua dataset.

⸻

🚀 Key Features
	•	Models Supported
	•	mT5-xl (fine-tuned with QLoRA in 4-bit)
	•	mBART50
	•	Llama-3.2-1B-Instruct (for baseline comparison)
	•	Experiments
	•	Multilingual summarization: English → English, French → French
	•	Cross-lingual summarization: French → English
	•	Zero-shot, 1-shot, and 2-shot settings
	•	Fine-tuning
	•	QLoRA fine-tuning for mT5
	•	Standard fine-tuning for mBART50
	•	Evaluation Metrics
	•	ROUGE-1, ROUGE-2, ROUGE-L
	•	BERTScore (F1)
	•	Reproducible Pipelines
	•	Dataset preprocessing and sampling
	•	Fine-tuning scripts with Hugging Face integration
	•	Automated evaluation and logging of metrics

⸻

📂 Repository Structure

├── download_model.py        # Utilities to load Hugging Face models
├── download_datasets.py     # Prepares WikiLingua dataset (EN, FR, FR→EN)
├── finetune.py              # Fine-tuning pipeline (QLoRA for mT5, standard for mBART50)
├── load_finetuned.py        # Load fine-tuned models from Hugging Face Hub
├── metrics.py               # ROUGE + BERTScore evaluation
├── main.py                  # Run experiments (zero-shot, 1-shot, 2-shot)
├── datasets/                # Preprocessed CSV datasets
│   ├── train.csv / val.csv / test.csv
│   ├── train_fr.csv / ...
│   └── train_cross.csv / ...
└── rouge_results.csv        # Evaluation results are stored here


⸻

📊 Results (Summary)
	•	Fine-tuning improves mBART50 and mT5 performance, but cross-lingual generalization remains weak.
	•	mBART50-English outperforms mBART50-Multilingual when evaluated on English.
	•	GPT-4o-mini shows strong zero-shot performance and benefits from 1-shot prompts, but gains plateau with 2-shot prompting.
	•	ICL is not inherent to transformers, but emerges in modern LLMs.

⸻

⚡️ Quickstart

1️⃣ Install dependencies

pip install -r requirements.txt

2️⃣ Download datasets

python download_datasets.py

3️⃣ Fine-tune a model

Example: Fine-tune mT5 on English summarization

python finetune.py --model mT5 --finetune_type english

Fine-tuned models will be pushed to your Hugging Face Hub.

4️⃣ Run experiments

Zero-shot evaluation with mBART50 on multilingual data:

python main.py --model mBART50 --experiment zero-shot --finetune_type multilingual --num_examples 100

1-shot evaluation with mT5 on cross-lingual data:

python main.py --model mT5 --experiment 1-shot --finetune_type crosslingual


⸻

📈 Evaluation

Scores are saved to rouge_results.csv in the format:

model_name, experiment_type, dataset_name, ROUGE-1, ROUGE-2, ROUGE-L, BERT-F1

You can analyze results with:

import pandas as pd
df = pd.read_csv("rouge_results.csv")
print(df.groupby(["model_name", "experiment_type"]).mean())


⸻

🤝 Contributing

Pull requests are welcome! Please open an issue first for discussions on major changes.

⸻

📜 Citation

If you use this code, please cite:

@article{aswal2025crosslingual,
  title={Exploring Cross Lingual Transfer Learning Techniques for Abstractive Text Summarization},
  author={Aswal, Darpan},
  journal={Preprint},
  year={2025}
}


⸻

🌟 Acknowledgments
	•	Hugging Face Transformers
	•	WikiLingua Dataset
	•	Inspiration from recent research on ICL emergence in LLMs

⸻

👉 This README is written to be developer-friendly while still highlighting the research contributions.

Would you like me to also create a visual diagram (pipeline illustration in Markdown/mermaid) for the README so people can quickly grasp your experiment workflow?
