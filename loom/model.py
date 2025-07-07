"""Wrapper around GPT-2 model for text generation."""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from transformers import TextIteratorStreamer
import threading
import os


class LoomModel:
    """Load GPT-2 XL model and generate text."""

    def __init__(
        self,
        model_name: str = "sshleifer/tiny-gpt2",
        device: str | None = None,
    ) -> None:
        # Allow overriding the device via environment variable or parameter. If
        # none is provided, pick the best available device including Apple's MPS
        # backend.
        if device is None:
            device = os.getenv("LOOM_DEVICE")

        # Validate and fall back to CPU if the requested device is unavailable.
        if device in {"cuda", "cuda:0", "gpu"}:
            if not torch.cuda.is_available():
                device = "cpu"
            else:
                device = "cuda"
        elif device == "mps":
            if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
                device = "cpu"
        elif device not in {None, "cpu"}:
            # Unknown device string
            device = "cpu"
        
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
                
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # Truncate long prompts from the beginning to fit the model context
        self.tokenizer.truncation_side = "left"
        # Load the model with reduced memory requirements. When running on a GPU
        # or MPS device we also switch to half precision to keep memory usage
        # low and avoid crashes on machines with limited RAM.
        dtype = torch.float16 if self.device != "cpu" else None
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )
        self.model.to(self.device)

    def _encode_prompt(
        self, prompt: str, *, room_for: int = 0
    ) -> torch.Tensor:
        """Tokenize ``prompt`` dropping earliest tokens if too long.

        Parameters
        ----------
        prompt:
            Text prompt to encode.
        room_for:
            Reserve this many tokens in the context for generation. The prompt
            will be truncated to ``model_max_length - room_for`` tokens if
            necessary.
        """
        max_len = max(1, self.tokenizer.model_max_length - room_for)
        return (
            self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=max_len,
                truncation=True,
            ).to(self.device)
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        num_return_sequences: int = 1,
        temperature: float = 0.8,
    ) -> list[str]:
        """Generate text from a prompt.

        Parameters
        ----------
        prompt:
            The text prompt to continue.
        max_new_tokens:
            Number of tokens to generate beyond the prompt.
        num_return_sequences:
            How many candidate continuations to produce.
        temperature:
            Sampling temperature controlling randomness.
        """
        inputs = self._encode_prompt(prompt, room_for=max_new_tokens)
        attention = torch.ones_like(inputs)
        with torch.inference_mode():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return [
            self.tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]

    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
    ):
        """Yield generated tokens one by one."""
        inputs = self._encode_prompt(prompt, room_for=max_new_tokens)
        attention = torch.ones_like(inputs)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        def _generate() -> None:
            with torch.inference_mode():
                self.model.generate(
                    input_ids=inputs,
                    attention_mask=attention,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    streamer=streamer,
                )

        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        for token in streamer:
            yield token
        thread.join()
