# src/openCHA/llms/llama.py

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

            # Always use the literal default model name here
            model_name = values.get(
                "model_name",
                "meta-llama/Llama-3.1-8B-Instruct"
            )
            hf_token = HfFolder.get_token()

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(
                    torch.float16
                    if torch.cuda.is_available()
                    else torch.float32
                ),
                device_map="auto",
                token=hf_token
            )

            # Populate both model and llm_model for downstream use
            values["tokenizer"] = tokenizer
            values["model"]     = model
            values["llm_model"] = model

        except ImportError as e:
            raise ValueError(
                "Missing dependencies. Run `pip install transformers huggingface_hub`"
            ) from e

        return values

    def get_model_names(self) -> List[str]:
        return list(self.models.keys())

    def is_max_token(self, model_name: str, query: str) -> bool:
        token_count = len(self.tokenizer.encode(query))
        return token_count > self.models.get(model_name, 0)

    def _prepare_prompt(self, prompt: str) -> str:
        # Official Instruct‑style wrapper for LLaMA 3
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{prompt}\n\n"
            "### Response:\n"
        )

    def _parse_response(self, response: str) -> str:
        # Clean up whitespace
        return response.strip()

    def generate(self, query: str, **kwargs: Any) -> str:
        prompt = self._prepare_prompt(query)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        raw = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        # Return only the model's answer portion
        answer = raw.split("### Response:")[-1].strip()
        return answer
