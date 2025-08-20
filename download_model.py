import torch
import huggingface_hub
from transformers import (AutoTokenizer, 
                          BitsAndBytesConfig, 
                          MBart50TokenizerFast,
                          AutoModelForSeq2SeqLM,
                          MBartForConditionalGeneration)
from config import hf_token
from huggingface_hub import login

device = "cuda" if torch.cuda.is_available() else "cpu"

def download_model(model_name: str):
    """Downloads the specified model."""
    if model_name == "mT5":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-xl",
                                                      quantization_config=bnb_config,
                                                      device_map="auto").to(device)
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-xl")
        return model, tokenizer
    elif model_name == "mBART50":
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50").to(device)
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="en_XX")
        return model, tokenizer
    elif model_name == "Llama-3.2-1B-Instruct":
        login(token = hf_token)
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to(device)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        return model, tokenizer
    else:
        raise ValueError("Invalid model name. Choose from 'mT5', 'mBART', 'Llama'.")