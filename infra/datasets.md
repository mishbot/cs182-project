# Refusal-direction datasets

The default local seeder prepares **AdvBench, MaliciousInstruct, TDC2023, and
Alpaca** and publishes them as one versioned corpus in the existing datasets
bucket. No model or GPU is needed.

```bash
source infra/scripts/aws-login.sh
DATASET_BUCKET=$(tofu -chdir=infra/persistent output -raw dataset_bucket)
uv run --locked --extra seed python infra/scripts/seed_dataset.py \
  --bucket "$DATASET_BUCKET"
```

The final stdout line is the corpus S3 URI. Downloads are cached in
`.cache/datasets` by default. Public source downloads total only tens of MiB;
seeding does not download model weights. GitHub API rate limits can be raised
using an existing `GITHUB_TOKEN` or `GH_TOKEN`; tokens are not recorded in
provenance or sent to raw-file URLs.

## Upstream-derived dataset definitions

The source selection and instruction preprocessing definitions are adapted from
[andyrdt/refusal_direction, dataset/generate_datasets.ipynb](https://github.com/andyrdt/refusal_direction/blob/9d852fae1a9121c78b29142de733cb1340770cc3/dataset/generate_datasets.ipynb),
revision `9d852fae1a9121c78b29142de733cb1340770cc3`. This provenance is stated in
the implementation, each corpus manifest, and the generated README. The
upstream Apache-2.0 license and a modification/attribution notice are packaged
with the library and copied into each corpus under `attribution/`.

| Output | Original publisher/source | Definition followed |
| --- | --- | --- |
| `advbench.parquet` | [llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks), `data/advbench/harmful_behaviors.csv` | Use the `goal` column, discard target responses |
| `maliciousinstruct.parquet` | [Princeton-SysML/Jailbreak_LLM](https://github.com/Princeton-SysML/Jailbreak_LLM), `data/MaliciousInstruct.txt` | One instruction per line |
| `tdc2023.parquet` | [centerforaisafety/tdc2023-starter-kit](https://github.com/centerforaisafety/tdc2023-starter-kit), red-teaming behavior lists | Concatenate development then test behaviors; do not use generated attack strings |
| `alpaca.parquet` | [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | Keep rows with empty `input` after stripping whitespace, then use `instruction` |

All selected prompts have surrounding whitespace stripped; empty prompts are
omitted. Source order and duplicates are preserved in the files. No upstream
responses, targets, contexts, categories, or formatted training text are retained.

Original dataset publishers retain their own terms. The andyrdt code license is
not a blanket dataset license. Available source licenses and Alpaca's dataset card
are included under `attribution/`; individual raw-file checksums and upstream
revisions are in the manifest. Alpaca's dataset card identifies CC BY-NC 4.0.

## Project-specific representation and sampling

Each Parquet has **exactly** these non-null columns:

| Column | Type | Value |
| --- | --- | --- |
| `prompt` | string | Processed instruction |
| `prompt_harm_label` | string | `harmful` for the three harmful sources; `unharmful` for Alpaca |
| `adversarial` | bool | `False` for all four sources |

These are the three input columns the project reads from WildGuardMix. The
activation pipeline derives its additional output columns itself. Harm labels
are experimental conventions: Alpaca has not been newly human-labeled or moderated.
`adversarial=False` distinguishes these direct instructions/behaviors from crafted
attack prompts; the name AdvBench does not make every goal an adversarial prompt.

The following choices belong to this project, not andyrdt's paper split recipe:

- Store every processed row; no 128-example source cap or seeding-time deduplication.
- Use approximately 80/20 train/validation membership with seed 0. Keep exact
  duplicate prompts together across all four sources and stratify by label and
  adversarial flag.
- Sample the training pool with equal harmful/harmless counts, without replacement.
  Harmful samples are pooled across sources, without per-source quotas. Odd sample
  sizes give the extra row to harmful examples.
- Exclude exact prompts with conflicting known labels during sampling, as before.
- Retain the existing strict refusal threshold `> 0` and non-adversarial analysis.

This is not a full reproduction of the paper's train/validation/test construction.
The current upstream sources yield 520 AdvBench, 100 MaliciousInstruct, 100
TDC2023, and 31,323 Alpaca instruction-only rows (32,043 total); actual counts and
revisions are always recorded per snapshot.

## Storage, versioning, and membership

```text
s3://DATASET_BUCKET/datasets/refusal-direction/SNAPSHOT_ID/
  advbench.parquet
  maliciousinstruct.parquet
  tdc2023.parquet
  alpaca.parquet
  README.md
  attribution/...
  artifact-manifest.json
```

The snapshot ID is a SHA-256 fingerprint of resolved GitHub/Hugging Face revisions,
raw-file checksums, preprocessing/version/attribution settings, and the Arrow writer.
It is not a single upstream Git SHA. Every source is downloaded at its resolved
immutable revision. The manifest lists file checksums and ordered source offsets.

`source_row_index` refers to the fixed concatenation order listed above. Metadata
provides each source's offset and row count without adding a source column to the
Parquets. Default membership is published separately under
`partitions/refusal-direction/SNAPSHOT_ID/...json`; no train/validation Parquets
are duplicated. Membership binds to the exact corpus manifest and ordered files.

The seeder writes the data completion manifest last, then publishes/verifies the
default membership. Repeating an identical seed reuses its immutable data and
membership. If membership publication fails, rerun the seed command to complete
it. Changed sources or preprocessing produce a new snapshot; existing snapshots
are never rewritten. A failed data upload has no completion manifest.

For a different membership configuration, use the existing command and pass its
returned URI to collection:

```bash
uv run --locked --extra seed python infra/scripts/partition_dataset.py \
  --dataset-s3-uri "$DATASET_URI" --validation-fraction 0.2 --seed 7
```

## Collection and experiment workflow

After seeding, rebuild/publish the runtime so it contains the new dataset reader
and two-class sampler, then refresh discovery and prepare that image:

```bash
source infra/modal/env.sh
uv run --locked --extra modal python -m infra.modal.prepare_image
uv run --locked --extra modal python infra/scripts/run_qwen_experiment.py
```

Discovery now defaults to the latest completed refusal-direction corpus.
`DATASET_URI` remains a single URI; no environment-variable format changes are
needed. The experiment wrapper forwards the exact selected corpus to both
collections. Its default sample size of 1,000 needs 500 eligible harmful training
examples; reduce it if the actual source/partition pool is smaller.

The default split is training membership; `--split validation` uses the held-out
membership. This corpus has no project-defined `test` split. Default five-fold
analysis needs at least five harmful and five harmless examples. All rows are
non-adversarial, so omit `--include-adversarial`; the wrapper reports that mismatch
before starting jobs.

## Explicit WildGuardMix mode

The previous WildGuardMix archive format and pipeline remain available:

```bash
uv run --locked --extra seed python infra/scripts/seed_dataset.py \
  --bucket "$DATASET_BUCKET" --dataset wildguardmix --revision main
source infra/modal/env.sh --dataset wildguardmix
uv run --locked --extra modal python infra/scripts/run_qwen_experiment.py \
  --dataset wildguardmix --include-adversarial
```

`--revision` selects the WildGuardMix upstream ref only. WildGuardMix retains its
separate partition step, four-group sampler, and official test split. The default
seeder does not upload WildGuardMix or require gated access to it.

Seeding uses the existing datasets-bucket operator permissions; no new AWS
infrastructure is required. Small Parquet snapshots and membership incur ordinary
S3 request/storage charges. Automated verification uses fixture data and an
in-memory S3 adapter, without network or timing assertions.
