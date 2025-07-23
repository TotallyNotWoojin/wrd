from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import HfFolder
import torch

model_name = "meta-llama/Llama-3.1-8B-Instruct"
token = HfFolder.get_token()

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", token=token)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe("Write a poem about a dragon", max_new_tokens=100)[0]['generated_text'])
