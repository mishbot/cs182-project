"""Exercise generation through the real original-Qwen source and tokenizer."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

from cs182_project import inference


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in
                       ["torch", "transformers", "tiktoken", "einops"]), "Install runtime dependencies")
class RealInferenceTest(unittest.TestCase):
    def test_tiny_original_qwen_generates_on_cpu(self):
        from test_activations import tiny_qwen

        model, tokenizer = tiny_qwen()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            tokenizer.save_pretrained(directory)
            original = (Path(directory)/"config.json").read_bytes()
            result = inference.generate(directory, "hello", 4, "cpu")
            self.assertEqual((Path(directory)/"config.json").read_bytes(), original)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
