"""Deterministic projection geometry and a tiny tied-Qwen S3 round trip."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from cs182_project import model_artifacts as artifacts
from cs182_project import weight_orthogonalization as orthogonalization
from test_activations import MemoryS3, tiny_qwen

AVAILABLE = all(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'numpy', 'safetensors'))
if AVAILABLE:
    import numpy as np
    import torch
    from cs182_project.qwen_model import load_model


class AnalysisS3(MemoryS3):
    def __init__(self):
        super().__init__()
        self.pages = []

    def get_paginator(self, name):
        return self

    def paginate(self, **kwargs):
        return iter(self.pages)


def analysis_snapshot(s3, analysis_id, model_checksum, direction):
    prefix = f'runs/activation/{analysis_id}'
    metrics = dict(manifest_sha256='a'*64, source_manifest=dict(model=dict(manifest_sha256=model_checksum)),
                   selected_directions=[dict(name='non_adversarial_midpoint', layer=1, valid=True)])
    buffer = io.BytesIO()
    np.savez_compressed(buffer, selected_names=np.array(['non_adversarial_midpoint']),
                        selected_layers=np.array([1]), selected_valid=np.array([True]),
                        selected_directions=direction[None, :])
    files = {'metrics.json': json.dumps(metrics).encode(), 'directions.npz': buffer.getvalue()}
    entries = []
    for filename, data in files.items():
        key = f'{prefix}/{filename}'
        s3.objects['analyses', key] = data
        entries.append(dict(key=key, size=len(data), sha256=hashlib.sha256(data).hexdigest()))
    manifest = dict(schema_version=1, status='complete', analysis_id=analysis_id,
                    source_manifest_sha256='a'*64, artifacts=entries)
    key = f'{prefix}/manifest.json'
    s3.objects['analyses', key] = json.dumps(manifest).encode()
    return key


@unittest.skipUnless(AVAILABLE, 'Install the runtime extra')
class WeightOrthogonalizationTest(unittest.TestCase):
    def test_rank_one_projection_matches_explicit_matrix_for_both_storage_orientations(self):
        rhat = torch.tensor([1., 2., -3.])
        rhat /= rhat.norm()
        projector = torch.eye(3) - torch.outer(rhat, rhat)
        for embedding, original in [(False, torch.arange(21, dtype=torch.float32).reshape(3, 7)),
                                    (True, torch.arange(21, dtype=torch.float32).reshape(7, 3))]:
            weight = original.clone()
            orthogonalization.project_weight(weight, rhat, embedding=embedding, chunk_size=2)
            expected = original @ projector if embedding else projector @ original
            torch.testing.assert_close(weight, expected, atol=4e-6, rtol=1e-5)
            residual = weight @ rhat if embedding else rhat @ weight
            torch.testing.assert_close(residual, torch.zeros_like(residual), atol=4e-6, rtol=0)
            projected = weight.clone()
            orthogonalization.project_weight(weight, rhat, embedding=embedding, chunk_size=2)
            torch.testing.assert_close(weight, projected, atol=4e-6, rtol=1e-5)
        # The saved dtype's rounding, rather than the rank-one arithmetic, limits precision.
        low_precision = torch.arange(21, dtype=torch.float32).reshape(7, 3).to(torch.bfloat16)
        expected = (low_precision.float() @ projector).to(torch.bfloat16)
        orthogonalization.project_weight(low_precision, rhat, embedding=True, chunk_size=2)
        self.assertEqual(low_precision.dtype, torch.bfloat16)
        torch.testing.assert_close(low_precision, expected)

    def test_latest_matching_analysis_and_tied_model_round_trip(self):
        s3 = AnalysisS3()
        model, tokenizer = tiny_qwen()
        model.config.tie_word_embeddings = True
        model.tie_weights()
        original = {name: value.detach().clone() for name, value in model.state_dict().items()}
        direction = np.arange(1, model.config.hidden_size + 1, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source'
            model.save_pretrained(source)
            tokenizer.save_pretrained(source)
            (source/'LICENSE').write_text('preserved license')
            # Existing checkpoints may contain an index superseded by save_pretrained.
            (source/'pytorch_model.bin.index.json').write_text('{}')
            files = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
            s3.snapshot('models', 'models/Qwen-1_8B-Chat/base', files,
                        model_id=artifacts.MODEL_ID, revision=artifacts.MODEL_REVISION, model_format="legacy_qwen")
            source_uri = 's3://models/models/Qwen-1_8B-Chat/base'
            manifest_bytes = s3.objects['models', 'models/Qwen-1_8B-Chat/base/'+artifacts.MANIFEST]
            checksum = hashlib.sha256(manifest_bytes).hexdigest()
            matching = analysis_snapshot(s3, 'matching', checksum, direction)
            other = analysis_snapshot(s3, 'other', '0'*64, -direction)
            s3.objects['analyses', 'runs/activation/incomplete/manifest.json'] = b'{"status":"incomplete"}'
            s3.pages = [dict(Contents=[dict(Key=matching, LastModified=datetime(2026, 11, 1, tzinfo=timezone.utc))]),
                        dict(Contents=[dict(Key=other, LastModified=datetime(2026, 11, 2, tzinfo=timezone.utc)),
                                       dict(Key='runs/activation/incomplete/manifest.json',
                                            LastModified=datetime(2026, 11, 3, tzinfo=timezone.utc))])]
            uri = orthogonalization.orthogonalize_model(
                s3, model_s3_uri=source_uri, analysis_root_s3_uri='s3://analyses/runs',
                device='cpu', cache_dir=root/'cache')
            self.assertRegex(uri, r'^s3://models/models/Qwen-1_8B-Chat-diff_of_means_max_f1-[0-9a-f]{32}$')
            self.assertEqual(s3.uploads[-1][1], uri.removeprefix('s3://models/')+'/'+artifacts.MANIFEST)
            directory = artifacts.download_model(s3, uri, root/'derived-cache')
            result = load_model(directory, "cpu")
            self.assertFalse(result.config.tie_word_embeddings)
            self.assertNotEqual(result.lm_head.weight.data_ptr(), result.transformer.wte.weight.data_ptr())
            manifest = json.loads((directory/artifacts.MANIFEST).read_text())
            self.assertEqual(manifest['direction']['analysis_id'], 'matching')
            self.assertEqual(manifest['source_model']['manifest_sha256'], checksum)
            self.assertEqual((directory/'LICENSE').read_bytes(), b'preserved license')
            self.assertFalse((directory/'pytorch_model.bin.index.json').exists())
            rhat = torch.as_tensor(direction, dtype=torch.float32)
            projector = torch.eye(rhat.numel()) - torch.outer(rhat, rhat)
            modified = {entry['name'] for entry in manifest['transformation']['modified_parameters']}
            self.assertEqual(len(modified), 1 + 2*model.config.num_hidden_layers)
            for name, weight in result.state_dict().items():
                expected = (original[name] @ projector if name == 'transformer.wte.weight'
                            else projector @ original[name] if name in modified else original[name])
                if name in modified:
                    torch.testing.assert_close(weight, expected, atol=1e-7, rtol=1e-5)
                else:
                    self.assertTrue(torch.equal(weight, expected), name)
            with torch.inference_mode():
                output = result(input_ids=torch.tensor([[3, 4]]))
            self.assertTrue(torch.isfinite(output.logits).all())

    def test_interrupted_model_publication_cannot_publish_a_completion_marker(self):
        s3 = AnalysisS3()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors'):
                (directory/name).write_bytes(b'fixture')
            s3.fail_upload_after = 1
            with self.assertRaisesRegex(RuntimeError, 'Incomplete model'):
                orthogonalization.publish_model(s3, directory, 'models', 'derived-fixture', 'fixture', {})
            self.assertFalse(any(key.endswith(artifacts.MANIFEST) for _, key in s3.objects))


if __name__ == '__main__':
    unittest.main()
