import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
import gc


def get_hf_llm(model_name="microsoft/phi-2", max_new_token=512, **kwargs):
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True,
        max_memory={"cpu": "4GB"}
    )
    
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        temperature=kwargs.get("temperature", 0.7),
        device= -1,
        do_sample=True,
        repetition_penalty=1.1,
        batch_size=1, 
        return_full_text=False  
    )
    
    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
        model_kwargs={
            "max_length": max_new_token + 100,  
            "pad_token_id": tokenizer.eos_token_id
        }
    )
    
    return llm
        

def get_lightweight_llm():
    return get_hf_llm("microsoft/DialoGPT-small", max_new_token=256, temperature=0.7)