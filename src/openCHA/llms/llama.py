from typing import Any, Dict, List
from openCHA.llms import BaseLLM
from pydantic import model_validator

class LlamaLLM(BaseLLM):
    """
    Meta LLaMA 3.1 8B Instruct using Hugging Face Transformers.
    """

    models: Dict = {
        "meta-llama/Llama-3.1-8B-Instruct": 8192
    }
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer: Any = None
    model: Any = None

    @model_validator(mode="before")
    def validate_environment(cls, values: Dict) -> Dict:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from huggingface_hub import HfFolder
            import torch

            model_name = values.get("model_name", "meta-llama/Llama-3.1-8B-Instruct")
            token = HfFolder.get_token()

            tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                token=token
            )

            values["tokenizer"] = tokenizer
            values["model"] = model
            values["model_name"] = model_name
        except ImportError as e:
            raise ValueError("Transformers or Hugging Face Hub not installed. Run `pip install transformers huggingface_hub`") from e
        return values

    def get_model_names(self) -> List[str]:
        return list(self.models.keys())

    def is_max_token(self, model_name, query) -> bool:
        token_count = len(self.tokenizer.encode(query))
        return token_count > self.models.get(model_name, 8192)

    def _parse_response(self, response: str) -> str:
        return response.strip()

    def _prepare_prompt(self, prompt: str) -> str:
        # You can update this with chat-style formatting if needed
        return prompt

    def generate(self, query: str, **kwargs: Any) -> str:
        inputs = self.tokenizer(self._prepare_prompt(query), return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_response(response)
