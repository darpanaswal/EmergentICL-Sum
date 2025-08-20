import os
import csv
import torch
import argparse
import numpy as np
import pandas as pd
import huggingface_hub
from datasets import Dataset
from load_finetuned import load_model
from metrics import compute_scores, save_scores
from transformers import (AutoTokenizer, BitsAndBytesConfig, MBart50TokenizerFast,
                          AutoModelForSeq2SeqLM, AutoModelForCausalLM, Trainer,
                          MBartForConditionalGeneration, TrainingArguments,
                          DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "mT5": "mT5",
    "mBART50": "mBART50"
}
LANGUAGE_CODES = {
    "English": "en_XX",
    "French": "fr_XX"
}
EXPERIMENTS = {
    "mT5": ["zero-shot"],
    "mBART50": ["zero-shot", "1-shot"]
}

def summarize_text_mt5(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors="pt", 
                       max_length=512, truncation=True, 
                       padding=True).to(model.device)
    summary_ids = model.generate(input_ids = inputs.input_ids, 
                                 max_length=60, 
                                 num_beams=4, length_penalty=2.0, 
                                 early_stopping=True)
    summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    return summaries
    
def summarize_text_mbart50(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors="pt",
                       max_length=1024, truncation=True, 
                       padding=True).to(model.device)
    summary_ids = model.generate(inputs.input_ids, max_length=60,
                                 num_beams=4, length_penalty=2.0,
                                 early_stopping=True)
    summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    return summaries

def experiments(model_name, experiment_type, num_examples, finetune_type):
    """Runs an experiment with the given model and dataset."""
    print(f"Starting Experiment: on {model_name}")

    # Construct dataset paths dynamically
    train = pd.read_csv(os.path.join(BASE_DIR, "datasets/train.csv"))
    train_fr = pd.read_csv(os.path.join(BASE_DIR, "datasets/train_fr.csv"))
    train_cross = pd.read_csv(os.path.join(BASE_DIR, "datasets/train_cross.csv"))
    val = pd.read_csv(os.path.join(BASE_DIR, "datasets/val.csv"))
    val_fr = pd.read_csv(os.path.join(BASE_DIR, "datasets/val_fr.csv"))
    val_cross = pd.read_csv(os.path.join(BASE_DIR, "datasets/val_cross.csv"))
    test = pd.read_csv(os.path.join(BASE_DIR, "datasets/test.csv"))
    test_fr = pd.read_csv(os.path.join(BASE_DIR, "datasets/test_fr.csv"))
    test_cross = pd.read_csv(os.path.join(BASE_DIR, "datasets/test_cross.csv"))

    test = test.sample(num_examples) if num_examples else test
    test_fr = test_fr.sample(num_examples) if num_examples else test_fr
    test_cross = test_cross.sample(num_examples) if num_examples else test_cross

    model, tokenizer = load_model(model_name, finetune_type)
    print(f"Model {model_name} loaded successfully.")

    if model_name == "mT5":
        summarize_text = summarize_text_mt5
    elif model_name == "mBART50":
        summarize_text = summarize_text_mbart50

    if experiment_type == "zero-shot":
        run_zero_shot(model_name, model, tokenizer, summarize_text, test, test_fr, test_cross)
    elif experiment_type == "1-shot":
        run_1_shot(model_name, model, tokenizer, summarize_text, train, train_fr, train_cross, test, test_fr, test_cross)
    elif experiment_type == "2-shot":
        run_2_shot(model_name, model, tokenizer, summarize_text, train, train_fr, train_cross, test, test_fr, test_cross)
    else:
        raise ValueError("Invalid experiment type.")

def run_zero_shot(model_name, model, tokenizer, summarize_text, test, test_fr, test_cross, batch_size=16):
    print("Running Zero-Shot Evaluation...")
    
    for dataset, name in [(test, "English"), (test_fr, "French"), (test_cross, "Cross-lingual")]:
        if model_name == "mBART50":
            if name == "English":
                tokenizer.src_lang = "en_XX"
            else:
                tokenizer.src_lang = "fr_XX"
        prefix = "Summarize in English: " if name == "Cross-lingual" else "Summarize the text: "
        texts = [f"{prefix}{row['source']}\n Summary: " for _, row in dataset.iterrows()]
        reference_summaries = dataset["target"].tolist()

        # Process in batches
        generated_summaries = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}")
            batch_summaries = summarize_text(batch_texts, model, tokenizer)
            generated_summaries.extend(batch_summaries)

        scores = compute_scores(generated_summaries, reference_summaries)
        save_scores(scores, model_name, "zero-shot", name)
        print(f"{name} Scores:", scores)


def run_1_shot(model_name, model, tokenizer, summarize_text, train, train_fr, train_cross, test, test_fr, test_cross, batch_size=16):
    print("Running 1-Shot Evaluation...")

    for dataset, train_data, name in [(test, train, "English"), (test_fr, train_fr, "French"), (test_cross, train_cross, "Cross-lingual")]:
        if model_name == "mBART50":
            if name == "English":
                tokenizer.src_lang = "en_XX"
            else:
                tokenizer.src_lang = "fr_XX"
        generated_summaries = []
        reference_summaries = []

        texts = []
        for _, sample in dataset.iterrows():
            one_shot = train_data.sample(1)
            source = one_shot["source"].iloc[0] 
            target = one_shot["target"].iloc[0]
            prefix = "Summarize in English: " if name == "Cross-lingual" else "Summarize the text: "
            prompt = (
                f"{prefix}{source}\n Summary: {target}\n\n"
                f"{prefix}{sample['source']}\n Summary: "
            )
            texts.append(prompt)
            reference_summaries.append(sample["target"])

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}")
            batch_summaries = summarize_text(batch_texts, model, tokenizer)
            generated_summaries.extend(batch_summaries)

        scores = compute_scores(generated_summaries, reference_summaries)
        save_scores(scores, model_name, "1-shot", name)
        print(f"{name} Scores:", scores)

def run_2_shot(model_name, model, tokenizer, summarize_text, train, train_fr, train_cross, test, test_fr, test_cross, batch_size=16):
    print("Running 2-Shot Evaluation...")

    for dataset, train_data, name in [(test, train, "English"), (test_fr, train_fr, "French"), (test_cross, train_cross, "Cross-lingual")]:
        if model_name == "mBART50":
            if name == "English":
                tokenizer.src_lang = "en_XX"
            else:
                tokenizer.src_lang = "fr_XX"
        generated_summaries = []
        reference_summaries = []

        texts = []
        for _, sample in dataset.iterrows():
            two_shots = train_data.sample(2)
            two_shot1, two_shot2 = two_shots.iloc[0], two_shots.iloc[1]
            source1, source2 = two_shot1["source"].iloc[0] , two_shot2["source"].iloc[0] 
            target1, target2 = two_shot1["target"].iloc[0] , two_shot2["target"].iloc[0] 
            prefix = "Summarize in English: " if name == "Cross-lingual" else "Summarize the text: "
            prompt = (
                f"{prefix}{two_shot1['source']}\n Summary: {two_shot1['target']}\n\n"
                f"{prefix}{two_shot2['source']}\n Summary: {two_shot2['target']}\n\n"
                f"{prefix}{sample['source']}\n Summary: "
            )
            texts.append(prompt)
            reference_summaries.append(sample["target"])

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_summaries = summarize_text(batch_texts, model, tokenizer)
            print(f"Processing batch {i//batch_size + 1}")
            generated_summaries.extend(batch_summaries)

        scores = compute_scores(generated_summaries, reference_summaries)
        save_scores(scores, model_name, "2-shot", name)
        print(f"{name} Scores:", scores)

        
def main():
    parser = argparse.ArgumentParser(description="Run experiments with different models.")
    parser.add_argument("--model", type=str, required=True, choices=MODELS.values(), help="The model to use.")
    parser.add_argument("--experiment", type=str, required=True, choices=sum(EXPERIMENTS.values(), []), help="The experiment to run.")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of examples to generate summaries on (optional).")
    parser.add_argument("--finetune_type", type=str, required=True, choices=["english", "multilingual", "crosslingual"], help="The type of fine-tuning to apply.")
    args = parser.parse_args()

    experiments(args.model, args.experiment, args.num_examples, args.finetune_type)

if __name__ == "__main__":
    main()