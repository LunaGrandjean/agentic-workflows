"""Standalone handler copied to the model artifact for the Hugging Face serving image."""

from pathlib import Path
from typing import Any


class EndpointHandler:
    """Load a Phi-3 model or LoRA adapter and implement the Vertex prediction contract."""

    def __init__(self, model_dir: str = ".") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        options = {
            "device_map": {"": device},
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "attn_implementation": "eager",
        }
        if Path(model_dir, "adapter_config.json").is_file():
            from peft import PeftConfig, PeftModel

            config = PeftConfig.from_pretrained(model_dir)
            base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, **options)
            self.model = PeftModel.from_pretrained(base, model_dir)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, **options)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.use_cache = True
        self.model.eval()

    @staticmethod
    def validate_request(data: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        if not isinstance(data, dict):
            raise ValueError("The request body must be a JSON object.")
        instances = data.get("instances")
        if not isinstance(instances, list) or not 1 <= len(instances) <= 8:
            raise ValueError("instances must contain between 1 and 8 prompts.")
        prompts = []
        for instance in instances:
            prompt = instance.get("prompt") if isinstance(instance, dict) else None
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Each instance must contain a nonempty prompt string.")
            prompts.append(prompt)
        supplied = data.get("parameters", {})
        if not isinstance(supplied, dict):
            raise ValueError("parameters must be a JSON object.")
        allowed = {"max_new_tokens", "do_sample", "temperature", "top_p", "top_k", "repetition_penalty"}
        if unknown := set(supplied) - allowed:
            raise ValueError(f"Unsupported generation parameters: {sorted(unknown)}")
        parameters = {"max_new_tokens": 64, "do_sample": False, **supplied}
        if type(parameters["max_new_tokens"]) is not int or not 1 <= parameters["max_new_tokens"] <= 256:
            raise ValueError("max_new_tokens must be an integer between 1 and 256.")
        if type(parameters["do_sample"]) is not bool:
            raise ValueError("do_sample must be a boolean.")
        for name in ("temperature", "repetition_penalty"):
            if name in parameters and (type(parameters[name]) not in (int, float) or not 0 < parameters[name] <= 10):
                raise ValueError(f"{name} must be positive and at most 10.")
        if "top_p" in parameters and (type(parameters["top_p"]) not in (int, float) or not 0 < parameters["top_p"] <= 1):
            raise ValueError("top_p must be in (0, 1].")
        if "top_k" in parameters and (type(parameters["top_k"]) is not int or parameters["top_k"] < 0):
            raise ValueError("top_k must be a nonnegative integer.")
        return prompts, parameters

    def __call__(self, data: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
        prompts, parameters = self.validate_request(data)
        predictions = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            input_length = inputs["input_ids"].shape[1]
            if input_length > 1024:
                raise ValueError("A prompt may contain at most 1024 tokens.")
            with self.torch.inference_mode():
                output = self.model.generate(
                    **inputs, **parameters, pad_token_id=self.tokenizer.pad_token_id,
                )
            response = self.tokenizer.decode(output[0, input_length:], skip_special_tokens=True).strip()
            predictions.append({"generated_text": response})
        return {"predictions": predictions}
