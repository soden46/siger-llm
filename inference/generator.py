# inference/generator.py
import torch
from typing import Iterator, Optional
from tokenizer.tokenizer import MultilingualTokenizer
from model.siger_model   import SigerLM

from inference.sampler import Sampler
import inference.generator


class Generator:
    """
    Autoregressive text generator.
    
    Cara kerja:
    Input tokens → model → logits → sample → append → repeat
    Kayak loop query SQL satu-satu, tiap iterasi dapat 1 token baru.
    """

    def __init__(
        self,
        model: SigerLM,
        tokenizer: MultilingualTokenizer,
        device: str = None,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int  = 200,
        temperature: float   = 0.8,
        top_k: int           = 50,
        top_p: float         = 0.9,
        repetition_penalty: float = 1.15,
        lang: Optional[str]  = None,     # "id", "en", "code"
        stop_tokens: list    = None,
    ) -> str:
        """
        Generate teks dari prompt.
        Return string lengkap (prompt + generated).
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(
            prompt, add_bos=True, lang=lang
        )
        generated_ids = list(input_ids)

        stop_tokens = stop_tokens or [self.tokenizer.eos_id]

        for _ in range(max_new_tokens):
            # Prepare input tensor
            x = torch.tensor([generated_ids], dtype=torch.long, device=self.device)

            # Forward pass
            logits, _ = self.model(x)

            # Ambil logits token terakhir saja
            next_logits = logits[0, -1, :]  # (vocab_size,)

            # Sample next token
            next_token = Sampler.sample(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                penalty=repetition_penalty,
                generated_ids=generated_ids,
                do_sample=(temperature > 0),
            )

            generated_ids.append(next_token)

            # Stop kalau ketemu EOS
            if next_token in stop_tokens:
                break

        # Decode — skip prompt, return generated only
        output_ids = generated_ids[len(input_ids):]
        return self.tokenizer.decode(output_ids, skip_special_tokens=False)

    @torch.inference_mode()
    def stream(
        self,
        prompt: str,
        max_new_tokens: int  = 200,
        temperature: float   = 0.8,
        top_k: int           = 50,
        top_p: float         = 0.9,
        repetition_penalty: float = 1.15,
        lang: Optional[str]  = None,
    ) -> Iterator[str]:
        """
        Streaming generation — yield 1 token per iterasi.
        Kayak Server-Sent Events di Laravel, token ngalir satu-satu.
        """
        input_ids     = self.tokenizer.encode(prompt, add_bos=True, lang=lang)
        generated_ids = list(input_ids)

        for _ in range(max_new_tokens):
            x = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
            logits, _ = self.model(x)
            next_logits = logits[0, -1, :]

            next_token = Sampler.sample(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                penalty=repetition_penalty,
                generated_ids=generated_ids,
            )

            if next_token == self.tokenizer.eos_id:
                break

            generated_ids.append(next_token)

            # Decode token baru aja (bukan seluruh sequence)
            token_str = self.tokenizer.decode([next_token], skip_special_tokens=True)
            yield token_str

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 200,
        **kwargs,
    ) -> list[str]:
        """Generate multiple prompts sekaligus (parallel)."""
        return [self.generate(p, max_new_tokens, **kwargs) for p in prompts]