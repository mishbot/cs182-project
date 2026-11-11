"""Exercise real Transformers/PyTorch with a tiny local model when installed."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import infer


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ["torch", "transformers", "tokenizers"]),
    "Run inside the built container to exercise the ML dependencies",
)
class RealInferenceTest(unittest.TestCase):
    def test_tiny_local_qwen_generates_on_cpu(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

        vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "hello": 3, "user": 4, "assistant": 5, ":": 6}
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]",
            model_input_names=["input_ids", "attention_mask"],
            chat_template="{% for message in messages %}{{ message['role'] + ': ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}",
        )
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=128, pad_token_id=0, eos_token_id=1, sliding_window=None,
        ))
        with tempfile.TemporaryDirectory() as directory:
            tokenizer.save_pretrained(directory)
            model.save_pretrained(directory, safe_serialization=True)
            result = infer.generate(directory, "hello", 4, "cpu")
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
