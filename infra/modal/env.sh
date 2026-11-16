#!/usr/bin/env bash
# Source after aws-login.sh to select the latest published inputs and runtime.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Use: source infra/modal/env.sh [--project PROJECT] [--region REGION]' >&2
  exit 2
fi

_cs182_modal_env() {
  local root discovered image model dataset output extra
  if [[ ${1-} == --help || ${1-} == -h ]]; then
    echo 'Use: source infra/modal/env.sh [--project PROJECT] [--region REGION]'
    echo 'Defaults: CS182_PROJECT or cs182-cddr; region and credentials from your AWS session.'
    return 0
  fi
  if ! command -v uv >/dev/null 2>&1; then
    echo 'Install uv before sourcing infra/modal/env.sh.' >&2
    return 1
  fi
  root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd) || return 1
  discovered=$(uv run --project "$root" --locked --extra modal python \
    "$root/infra/scripts/discover_modal.py" "$@") || return 1
  if [[ $discovered == *$'\n'* ]]; then
    echo 'Modal discovery returned unexpected multiline output; environment unchanged.' >&2
    return 1
  fi
  IFS=$'\t' read -r image model dataset output extra <<< "$discovered" || return 1
  if [[ -z $image || -z $model || -z $dataset || -z $output || -n $extra ]]; then
    echo 'Modal discovery returned incomplete configuration; environment unchanged.' >&2
    return 1
  fi
  # Assign literal values, without evaluating AWS-provided text as shell code.
  export CS182_MODAL_IMAGE="$image" MODEL_URI="$model" DATASET_URI="$dataset" OUTPUT_URI="$output"
  printf 'CS182_MODAL_IMAGE=%s\nMODEL_URI=%s\nDATASET_URI=%s\nOUTPUT_URI=%s\n' \
    "$CS182_MODAL_IMAGE" "$MODEL_URI" "$DATASET_URI" "$OUTPUT_URI"
}

if _cs182_modal_env "$@"; then
  unset -f _cs182_modal_env
else
  unset -f _cs182_modal_env
  return 1
fi
