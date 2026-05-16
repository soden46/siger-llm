# optimization/onnx/export.py
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path


class ONNXExporter:
    """
    Export model ke ONNX → jalanin via ONNXRuntime.
    ONNXRuntime di CPU bisa 2-3x lebih cepat dari PyTorch CPU.
    Ini yang dipake production buat CPU deployment.
    """

    def __init__(self, model, save_dir: str = "./checkpoints/onnx"):
        self.model    = model.cpu().eval()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        seq_len: int     = 64,
        batch_size: int  = 1,
        opset: int       = 17,          # ONNX opset terbaru
    ) -> str:
        print("📦 Exporting to ONNX...")

        onnx_path = str(self.save_dir / "model.onnx")
        dummy_input = torch.randint(0, 1000, (batch_size, seq_len))

        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "logits":    {0: "batch", 1: "seq_len"},
            },
            opset_version=opset,
            do_constant_folding=True,   # fold constant ops = lebih cepat
        )

        # Verify
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX export valid: {onnx_path}")

        return onnx_path

    def optimize_onnx(self, onnx_path: str) -> str:
        """
        Optimasi graph ONNX:
        - Fuse operators
        - Eliminate redundant nodes
        - Constant folding
        """
        from onnxruntime.transformers import optimizer as ort_optimizer

        opt_path = onnx_path.replace(".onnx", "_optimized.onnx")

        opt_model = ort_optimizer.optimize_model(
            onnx_path,
            model_type="bert",        # paling dekat sama LLM
            num_heads=8,
            hidden_size=512,
            optimization_level=99,    # max optimization
        )
        opt_model.save_model_to_file(opt_path)
        print(f"✅ Optimized ONNX: {opt_path}")
        return opt_path

    def build_session(self, onnx_path: str) -> ort.InferenceSession:
        """
        Build ONNXRuntime session dengan CPU optimizations.
        """
        opts = ort.SessionOptions()

        # Threading — sesuai 2 core VPS lo
        opts.intra_op_num_threads = 2   # thread per operator
        opts.inter_op_num_threads = 1   # thread antar operator
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Graph optimization level (max)
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Enable memory pattern reuse
        opts.enable_mem_pattern    = True
        opts.enable_mem_reuse      = True
        opts.enable_cpu_mem_arena  = True

        session = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        print("✅ ONNX Runtime session ready")
        return session


class ONNXGenerator:
    """
    Generator yang pakai ONNX session instead of PyTorch.
    Drop-in replacement buat Generator class sebelumnya.
    """

    def __init__(self, session: ort.InferenceSession, tokenizer):
        self.session   = session
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        max_new_tokens: int  = 200,
        temperature: float   = 0.8,
        top_k: int           = 50,
        top_p: float         = 0.9,
    ) -> str:
        from inference.sampler import Sampler

        input_ids     = self.tokenizer.encode(prompt, add_bos=True)
        generated_ids = list(input_ids)

        for _ in range(max_new_tokens):
            x = np.array([generated_ids], dtype=np.int64)

            # ONNX inference — jauh lebih cepat dari PyTorch CPU
            outputs  = self.session.run(["logits"], {"input_ids": x})
            logits   = torch.tensor(outputs[0][0, -1, :])

            next_token = Sampler.sample(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generated_ids=generated_ids,
            )

            if next_token == self.tokenizer.eos_id:
                break
            generated_ids.append(next_token)

        output_ids = generated_ids[len(input_ids):]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)