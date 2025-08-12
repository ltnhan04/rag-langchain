import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

def get_hf_llm(model_name = "microsoft/phi-2", max_new_token = 1024, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=None)
    model_pipeline = pipeline(
        "text-generation",
        model = model,
        tokenizer=tokenizer,
        max_new_tokens = max_new_token,
        pad_token_id = tokenizer.eos_token_id,
        temperature = 0.7,
        device = -1
    )
    llm =  HuggingFacePipeline(
        pipeline=model_pipeline
    )
    return llm