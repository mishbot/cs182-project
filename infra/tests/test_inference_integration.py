"""Exercise real Transformers/PyTorch with a tiny local model when installed."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cs182_project import inference
from cs182_project._model_config import load_local_model_config


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
        # Mimic the pinned checkpoint: sliding-window attention is disabled,
        # but its config still carries a non-null window length.
        model.config.use_sliding_window = False
        model.config.sliding_window = 128
        with tempfile.TemporaryDirectory() as directory:
            tokenizer.save_pretrained(directory)
            model.save_pretrained(directory, safe_serialization=True)
            config_path = Path(directory) / "config.json"
            original = config_path.read_bytes()
            self.assertEqual(json.loads(original)["sliding_window"], 128)
            with patch("transformers.models.qwen2.modeling_qwen2.logger.warning_once") as warning:
                result = inference.generate(directory, "hello", 4, "cpu")
            warning.assert_not_called()
            self.assertEqual(config_path.read_bytes(), original)
        self.assertIsInstance(result, str)

    def test_enabled_sliding_window_is_preserved(self):
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

        with tempfile.TemporaryDirectory() as directory:
            Qwen2Config(use_sliding_window=True, sliding_window=16).save_pretrained(directory)
            config = load_local_model_config(directory)
        self.assertTrue(config.use_sliding_window)
        self.assertEqual(config.sliding_window, 16)


if __name__ == "__main__":
    unittest.main()
