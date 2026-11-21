"""CPU seeder with a stable import path for Modal serialization."""


def execute_seed(bucket, revision, prefix):
    import logging
    import tempfile
    import boto3
    from botocore.config import Config
    from cs182_project.seeding import seed_model

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with tempfile.TemporaryDirectory(prefix="cs182-modelscope-") as directory:
        return seed_model(boto3.client('s3', config=Config(retries={'mode': 'standard', 'max_attempts': 5})),
                          bucket, revision=revision, prefix=prefix or None, cache_dir=directory)
