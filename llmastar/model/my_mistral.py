from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class MyMistral:
    def __init__(self, prompt="standard"):
        model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

    def run(self, query: str) -> str:
        inputs = self.tokenizer(query, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
