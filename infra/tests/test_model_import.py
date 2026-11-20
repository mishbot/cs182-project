"""Small deterministic archival round trip and interrupted-publication check."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from cs182_project.model_artifacts import MANIFEST, download_model
from cs182_project.model_import import import_model
from test_activations import MemoryS3


class TransferS3(MemoryS3):
    def upload_file(self, filename, bucket, key, Config=None, ExtraArgs=None):
        with open(filename, 'rb') as body:
            self.upload_fileobj(body, bucket, key, ExtraArgs)

    def head_object(self, Bucket, Key):
        return {'ContentLength': len(self.objects[Bucket, Key])}


class ModelImportTest(unittest.TestCase):
    def fixture(self, root):
        files = {
            'config.json': b'{"model_type":"qwen","architectures":["QWenLMHeadModel"]}',
            'tokenizer_config.json': b'{"tokenizer_class":"QWenTokenizer"}',
            'qwen.tiktoken': b'original tokenizer data',
            'configuration_qwen.py': b'original config module',
            'modeling_qwen.py': b'original model module',
            'tokenization_qwen.py': b'original tokenizer module',
            'model.safetensors': b'original merged weights',
            'generation_config.json': b'{"use_cache":false}',
            'LICENSE': b'original license',
            'assets/NOTICE': b'nested original notice',
        }
        for name, data in files.items():
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return files

    def test_legacy_snapshot_preserves_files_and_round_trips_through_the_downloader(self):
        s3 = TransferS3()
        source = dict(workspace='mishbot', environment='main', volume='cs182-runs', path='/v1/merged')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'merged'
            files = self.fixture(root)
            uri = import_model(s3, root, 'models', source=source)
            self.assertRegex(uri, r'^s3://models/models/Qwen-1_8B-Chat-merged-[0-9a-f]{32}$')
            self.assertTrue(s3.uploads[-1][1].endswith('/'+MANIFEST))
            directory = download_model(s3, uri, Path(temporary)/'download')
            manifest = json.loads((directory/MANIFEST).read_text())
            self.assertEqual(manifest['source'], source)
            self.assertEqual(manifest['model_format'], 'legacy_qwen')
            self.assertEqual(manifest['artifact_kind'], 'modal_volume_import')
            self.assertEqual({entry['path'] for entry in manifest['files']}, set(files))
            for entry in manifest['files']:
                name = entry['path']
                self.assertEqual((directory/name).read_bytes(), files[name])
                self.assertEqual(entry['sha256'], hashlib.sha256(files[name]).hexdigest())

    def test_interrupted_import_does_not_publish_completion(self):
        s3 = TransferS3()
        s3.fail_upload_after = 2
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            with self.assertRaisesRegex(RuntimeError, 'Incomplete model'):
                import_model(s3, root, 'models', source={'kind': 'fixture'})
        self.assertTrue(s3.uploads)
        self.assertFalse(any(key.endswith('/'+MANIFEST) for _, key in s3.objects))


if __name__ == '__main__':
    unittest.main()
