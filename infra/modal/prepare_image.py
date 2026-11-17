"""Import and prepare the selected ECR image in Modal before running GPU tasks."""

import argparse
import logging
import os

from infra.modal._runner import aws_environment, modal_image_name, prepare_runtime_image


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    reference = os.environ.get("CS182_MODAL_IMAGE", "")
    modal_image_name(reference)  # Validate before resolving AWS credentials.
    import boto3

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    credentials = aws_environment(boto3.Session())
    print(prepare_runtime_image(reference, credentials))


if __name__ == "__main__":
    main()
