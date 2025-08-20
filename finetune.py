import os
import csv
import torch
import argparse
import numpy as np
import pandas as pd
import huggingface_hub
from config import hf_token
from datasets import Dataset
from huggingface_hub import login
from download_model import download_model
from huggingface_hub import HfApi, HfFolder
from transformers import (AutoTokenizer, BitsAndBytesConfig, MBart50TokenizerFast,
                          AutoModelForSeq2SeqLM, AutoModelForCausalLM, Trainer,
                          MBartForConditionalGeneration, TrainingArguments,
                          DataCollatorForSeq2Seq, EarlyStoppingCallback)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "mT5": "mT5",
    "mBART50": "mBART50"
}

def summarize_text_mt5(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors="pt", 
                       max_length=512, truncation=True, 
                       padding=True).to(model.device)
    summary_ids = model.generate(inputs.input_ids, 
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

    summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    return summaries

def experiments(model_name, finetune_type):
    login(token = hf_token)
    """Runs an experiment with the given model and dataset."""
    print(f"Starting Experiment: on {model_name}")

    train = pd.read_csv(os.path.join(BASE_DIR, "datasets/train.csv"))
    train_fr = pd.read_csv(os.path.join(BASE_DIR, "datasets/train_fr.csv"))
    train_cross = pd.read_csv(os.path.join(BASE_DIR, "datasets/train_cross.csv"))
    val = pd.read_csv(os.path.join(BASE_DIR, "datasets/val.csv"))
    val_fr = pd.read_csv(os.path.join(BASE_DIR, "datasets/val_fr.csv"))
    val_cross = pd.read_csv(os.path.join(BASE_DIR, "datasets/val_cross.csv"))
    test = pd.read_csv(os.path.join(BASE_DIR, "datasets/test.csv"))
    test_fr = pd.read_csv(os.path.join(BASE_DIR, "datasets/test_fr.csv"))
    test_cross = pd.read_csv(os.path.join(BASE_DIR, "datasets/test_cross.csv"))

    model, tokenizer = download_model(model_name)
    print(f"Model {model_name} loaded successfully.")

    if model_name == "mT5":
        summarize_text = summarize_text_mt5
    elif model_name == "mBART50":
        summarize_text = summarize_text_mbart50

    if finetune_type == "english":
        fine_tune(model_name, "english", model, tokenizer, summarize_text, train, val)
    elif finetune_type == "multilingual":
        fine_tune(model_name, "multilingual", model, tokenizer, summarize_text, train_fr, val_fr)
    else:
        fine_tune(model_name, "crosslingual", model, tokenizer, summarize_text, train_cross, val_cross)


def fine_tune(model_name, finetune_type, model, tokenizer, summarize_text, train, val):
    print("Starting Fine-tuning...")
    if model_name == "mT5":
        max_input = 512 
    else:
        max_input = 1024

    max_output = 60
    train_dataset = train
    eval_dataset = val
    if finetune_type == "english":
        train_dataset = Dataset.from_pandas(train.sample(1500))
        eval_dataset = Dataset.from_pandas(val.sample(200))
    else:
        train_dataset = Dataset.from_pandas(train.sample(1200))
        eval_dataset = Dataset.from_pandas(val.sample(150))
    def preprocess_function(examples):
        inputs = [f"Summarize the text: {ex}" for ex in examples["source"]]
        targets = [f"Summary: {ex}" for ex in examples["target"]]
        model_inputs = tokenizer(inputs, max_length=max_input, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_output, truncation=True)

        model_inputs["labels"] = labels["input_ids"]


        print("Input:", inputs[0])
        print("Output:", targets[0])

        return model_inputs

    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_eval = eval_dataset.map(preprocess_function, batched=True)

    # QLoRA config for mT5
    if model_name == "mT5":
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model) # Initialize the DataCollatorForSeq2Seq

    training_args = TrainingArguments(
        output_dir=f"./{model_name}-{finetune_type}-finetuned",
        evaluation_strategy="epoch",
        save_total_limit = 1,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=True,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )
    
    trainer.train()
    
    tokenizer.save_pretrained(training_args.output_dir)
    tokenizer.push_to_hub(f"{model_name}-{finetune_type}-finetuned")

    print("Saving model to Hugging Face Hub...")
    

def main():
    parser = argparse.ArgumentParser(description="Run experiments with different models.")
    parser.add_argument("--model", type=str, required=True, choices=MODELS.values(), help="The model to use.")
    parser.add_argument("--finetune_type", type=str, required=True, choices=["english", "multilingual", "crosslingual"], help="The type of fine-tuning to apply.")
    args = parser.parse_args()

    experiments(args.model, args.finetune_type)

if __name__ == "__main__":
    main()