"""Deterministic source → Parquet → S3 → grouped collection integration."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cs182_project import activations, dataset_artifacts as artifacts, dataset_partition as partition
from cs182_project import processed_datasets as corpus
from cs182_project.model_artifacts import MANIFEST
from test_activations import MemoryS3

AVAILABLE = importlib.util.find_spec('pyarrow') is not None


def source_fixtures(root):
    import pyarrow as pa
    import pyarrow.parquet as pq

    packages=[]
    for index, (name, repository, filenames) in enumerate(corpus.GITHUB_SOURCES):
        files={}
        for file_index, filename in enumerate(filenames):
            path=root/name/filename
            path.parent.mkdir(parents=True,exist_ok=True)
            if name=='advbench':
                path.write_text('goal,target\n'+''.join(f'harmful-{i},discard target\n' for i in range(12)))
            elif name=='maliciousinstruct':
                path.write_text('harmful-1\n'+'\n'.join(f'malicious-{i}' for i in range(8))+'\n')
            else:
                path.write_text(json.dumps([f'tdc-{file_index}-{i}' for i in range(5)]))
            files[filename]=path
        packages.append(corpus.DatasetSource(name,repository,str(index)*40,files))
    path=root/'alpaca.parquet'
    records=[dict(instruction=f'benign-{i}',input='',output='discard',text='discard') for i in range(24)]
    records += [dict(instruction='harmful-0',input='',output='discard',text='discard'),
                dict(instruction='needs context',input='context required',output='discard',text='discard')]
    pq.write_table(pa.Table.from_pylist(records),path)
    packages.append(corpus.DatasetSource('alpaca','tatsu-lab/alpaca','a'*40,{'data/train-fixture.parquet':path}))
    return packages


@unittest.skipUnless(AVAILABLE,'Install the seed extra')
class ProcessedCorpusTest(unittest.TestCase):
    def test_round_trip_partition_and_balanced_global_sampling(self):
        import pyarrow.parquet as pq

        s3=MemoryS3()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(corpus,'download_sources',return_value=source_fixtures(root)):
                uri=corpus.seed_corpus(s3,'datasets',root)
                uploads=len(s3.uploads)
                self.assertEqual(uri,corpus.seed_corpus(s3,'datasets',root))
                self.assertEqual(len(s3.uploads),uploads)
            local=artifacts.download_dataset(s3,uri,root/'cache')
            manifest=json.loads((local/MANIFEST).read_text())
            rows=artifacts.read_prompt_rows(local,manifest)
            for filename in artifacts.CORPUS_FILES:
                self.assertTrue(pq.read_schema(local/filename).equals(artifacts.prompt_schema()))
            alpaca=pq.read_table(local/'alpaca.parquet').to_pylist()
            self.assertEqual(len(alpaca),25)
            self.assertTrue(all(row['prompt_harm_label']=='unharmful' and row['adversarial'] is False for row in alpaca))
            self.assertNotIn('needs context',{row['prompt'] for row in rows})
            for item in manifest['sources']:
                self.assertEqual(item['name'],artifacts.CORPUS_FILES[manifest['sources'].index(item)].removesuffix('.parquet'))
                self.assertEqual(rows[item['row_offset']:item['row_offset']+item['row_count']],pq.read_table(local/item['file']).to_pylist())
            from cs182_project.model_artifacts import sha256
            membership,_=partition.load_partition(s3,partition.partition_uri(uri,manifest),manifest,sha256(local/MANIFEST),len(rows))
            train=set(membership['splits']['train'])
            validation=set(membership['splits']['validation'])
            self.assertFalse(train & validation)
            self.assertEqual(train | validation,set(range(len(rows))))
            duplicate=[index for index,row in enumerate(rows) if row['prompt']=='harmful-1']
            self.assertTrue(set(duplicate)<=train or set(duplicate)<=validation)
            selected=activations.sample_prompts(rows,10,0,train)
            self.assertEqual(sum(row['harmful'] for row in selected),5)
            self.assertTrue(all(not row['adversarial'] for row in selected))
            self.assertNotIn('harmful-0',{row['prompt'] for row in selected})
            self.assertTrue(all(rows[row['source_row_index']]['prompt']==row['prompt'] for row in selected))
            readme=(local/'README.md').read_text()
            self.assertIn(corpus.DEFINITIONS_URL,readme)
            self.assertIn('not andyrdt',readme)

    def test_interrupted_corpus_upload_has_no_completion_marker(self):
        s3=MemoryS3()
        s3.fail_upload_after=1
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(corpus,'download_sources',return_value=source_fixtures(Path(directory))):
                with self.assertRaisesRegex(OSError,'interrupted'):
                    corpus.seed_corpus(s3,'datasets',directory)
        self.assertTrue(s3.uploads)
        self.assertFalse(any(key.endswith('/'+MANIFEST) for _,key in s3.objects))


if __name__=='__main__':
    unittest.main()
