import torch
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
)
from peft import PeftModel

def load_model(model_name, finetune_type):
    """Loads a fine-tuned model from the Hugging Face repository based on its type."""
    if model_name not in MODEL_REPOS:
        raise ValueError(f"Invalid model name. Choose from: {list(MODEL_REPOS.keys())}")
    
    if finetune_type not in MODEL_REPOS[model_name]:
        raise ValueError(f"Invalid finetune type. Choose from: {list(MODEL_REPOS[model_name].keys())}")

    repo_name = MODEL_REPOS[model_name][finetune_type]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(repo_name)

    if model_name == "mT5":  # 4-bit quantized + QLoRA fine-tuned
        print(f"Loading {model_name} with {finetune_type} finetuning, 4-bit quantization, and QLoRA...")

        # Load model with 4-bit quantization settings
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        base_model_name = "google/mt5-xl"  # Use correct base model
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, quantization_config=quant_config, device_map="auto")

        # Apply fine-tuned LoRA adapter
        model = PeftModel.from_pretrained(model, repo_name)

    elif model_name == "mBART50":  # Normally fine-tuned
        print(f"Loading {model_name} with {finetune_type} fine-tuning...")

        model = AutoModelForSeq2SeqLM.from_pretrained(repo_name)
        model.to(device)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"{model_name} ({finetune_type}) loaded successfully!")
    return model, tokenizer

MODEL_REPOS = {
    "mT5": {
        "english": "darpanaswal/mT5-english-finetuned",
        "multilingual": "darpanaswal/mT5-multilingual-finetuned",
        "crosslingual": "darpanaswal/mT5-crosslingual-finetuned",
    },
    "mBART50": {
        "english": "darpanaswal/mBART50-english-finetuned",
        "multilingual": "darpanaswal/mBART50-multilingual-finetuned",
        "crosslingual": "darpanaswal/mBART50-crosslingual-finetuned",
    },
}