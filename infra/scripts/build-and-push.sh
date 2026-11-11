#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 || $1 == --help ]]; then
  echo "Usage: $0 REGION ECR_REPOSITORY_URL UNIQUE_IMAGE_TAG" >&2
  exit 2
fi
region=$1
repository=$2
tag=$3
if [[ ! $repository =~ ^[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[a-z0-9._/-]+$ || ! $tag =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}$ ]]; then
  echo 'Expected a private ECR repository URL and a valid image tag.' >&2
  exit 2
fi
registry=${repository%%/*}
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "$registry"
docker build --platform linux/amd64 -f "$root/infra/container/Dockerfile" -t "$repository:$tag" "$root"
docker push "$repository:$tag"
printf 'Published %s:%s\n' "$repository" "$tag"
