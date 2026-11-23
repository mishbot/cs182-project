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
cache_from=()
# BatchGetImage distinguishes an absent tag (a normal cold build) from API,
# repository, or permission errors. The service role already has this permission.
cache_status=$(aws ecr batch-get-image --region "$AWS_DEFAULT_REGION" \
  --repository-name "${CACHE_REPOSITORY_URI#*/}" --image-ids imageTag=buildcache \
  --query 'images[0].imageId.imageDigest || failures[0].failureCode' --output text)
case "$cache_status" in
  sha256:*)
    printf 'Using build cache %s:buildcache (%s)\n' "$CACHE_REPOSITORY_URI" "$cache_status"
    cache_from=(--cache-from "type=registry,ref=$CACHE_REPOSITORY_URI:buildcache")
    ;;
  ImageNotFound)
    printf 'No buildcache tag yet; starting a cold build and saving its cache before tests.\n'
    ;;
  *)
    printf 'Cannot read build cache: %s\n' "$cache_status" >&2
    exit 1
    ;;
esac
build_args=(
  --platform linux/amd64
  "${cache_from[@]}"
  --label "org.opencontainers.image.revision=$CODEBUILD_RESOLVED_SOURCE_VERSION"
  --label "org.opencontainers.image.source=$SOURCE_REPOSITORY_URL"
  -f infra/container/Dockerfile -t "$image"
)
# Persist the build graph before tests so later attempts can reuse dependency
# layers even if a test fails. Only the separate cache tag is published here.
docker buildx build "${build_args[@]}" --load \
  --cache-to "type=registry,ref=$CACHE_REPOSITORY_URI:buildcache,mode=max,image-manifest=true,oci-mediatypes=true,compression=estargz,force-compression=true" .

# Require the real ML dependencies; a missing dependency must not silently skip
# the tiny local model tests. No model download or AWS calls are needed.
docker run --rm --network none --entrypoint python "$image" \
  -c 'import torch, transformers, pyarrow, sklearn, plotly, cs182_project'
docker run --rm --network none --entrypoint python \
  -v "$PWD/infra:/test-source/infra:ro" "$image" \
  -m unittest discover -s /test-source/infra/tests -v
# Validate the shared CPU analysis / GPU inference environment.
docker run --rm --network none --entrypoint uv "$image" \
  pip check --python /app/.venv/bin/python

# Re-export the cached, tested build graph directly to the registry. Loading an
# image into Docker and using `docker push` would discard eStargz compression.
# Keep publication in this phase: CodeBuild post_build also runs on failure.
docker buildx build "${build_args[@]}" \
  --output "type=registry,compression=estargz,force-compression=true,oci-mediatypes=true" .
printf 'Published %s from %s\n' "$image" "$CODEBUILD_RESOLVED_SOURCE_VERSION"
