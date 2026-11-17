#!/usr/bin/env python3
"""Build and publish the clean checkout's GitHub HEAD using AWS CodeBuild."""

import argparse
import logging
from pathlib import Path
import sys

from cs182_project.builds import build_runtime_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("region")
    parser.add_argument("repository", help="Full private ECR repository URL")
    parser.add_argument("image_tag", help="Unique immutable image tag")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        import boto3

        print(build_runtime_image(boto3.Session(), args.region, args.repository, args.image_tag,
                                  Path(__file__).resolve().parents[2]))
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"Image build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
