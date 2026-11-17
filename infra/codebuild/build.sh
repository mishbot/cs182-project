#!/usr/bin/env bash
set -euo pipefail

cd "$CODEBUILD_SRC_DIR"
: "${RUNTIME_REPOSITORY_URI:?Set the runtime repository URI}"
: "${CACHE_REPOSITORY_URI:?Set the cache repository URI}"
: "${IMAGE_TAG:?Provide a unique image tag}"
: "${EXPECTED_SOURCE_VERSION:?Provide the exact GitHub commit}"
[[ $IMAGE_TAG =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}$ ]]
[[ $EXPECTED_SOURCE_VERSION =~ ^[0-9a-f]{40}$ ]]
[[ $(git rev-parse HEAD) == "$EXPECTED_SOURCE_VERSION" ]]
[[ $CODEBUILD_RESOLVED_SOURCE_VERSION == "$EXPECTED_SOURCE_VERSION" ]]

if ! docker info >/dev/null 2>&1; then
  nohup dockerd --host=unix:///var/run/docker.sock >/tmp/dockerd.log 2>&1 &
  timeout 60 bash -c 'until docker info >/dev/null 2>&1; do sleep 1; done'
fi
registry=${RUNTIME_REPOSITORY_URI%%/*}
aws ecr get-login-password --region "$AWS_DEFAULT_REGION" |
  docker login --username AWS --password-stdin "$registry"
docker buildx create --name runtime-builder --driver docker-container --use
docker buildx inspect --bootstrap
image="$RUNTIME_REPOSITORY_URI:$IMAGE_TAG"
docker buildx build --platform linux/amd64 --load \
  --cache-from "type=registry,ref=$CACHE_REPOSITORY_URI:buildcache" \
  --cache-to "type=registry,ref=$CACHE_REPOSITORY_URI:buildcache,mode=max,image-manifest=true,oci-mediatypes=true" \
  --label "org.opencontainers.image.revision=$CODEBUILD_RESOLVED_SOURCE_VERSION" \
  --label "org.opencontainers.image.source=$SOURCE_REPOSITORY_URL" \
  -f infra/container/Dockerfile -t "$image" .

# Require the real ML dependencies; a missing dependency must not silently skip
# the tiny local model tests. No model download or AWS calls are needed.
docker run --rm --network none --entrypoint python "$image" \
  -c 'import torch, transformers, pyarrow, cs182_project'
docker run --rm --network none --entrypoint python \
  -v "$PWD/infra:/test-infra:ro" "$image" \
  -m unittest discover -s /test-infra/tests -v

# Keep publication in this phase: CodeBuild post_build also runs on failure.
docker push "$image"
printf 'Published %s from %s\n' "$image" "$CODEBUILD_RESOLVED_SOURCE_VERSION"
