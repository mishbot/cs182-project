#!/usr/bin/env bash
# Source this file so the selected profile remains available in your terminal.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Use: source infra/scripts/aws-login.sh [--remote]' >&2
  exit 2
fi

_cs182_aws_login() {
  local version major minor
  if ! command -v aws >/dev/null 2>&1; then
    echo 'Install AWS CLI >= 2.32.0 before sourcing this file.' >&2
    return 1
  fi
  version=$(aws --version 2>&1) || return 1
  if [[ ! $version =~ aws-cli/([0-9]+)\.([0-9]+)\. ]]; then
    echo "Could not determine AWS CLI version: $version" >&2
    return 1
  fi
  major=${BASH_REMATCH[1]}
  minor=${BASH_REMATCH[2]}
  if (( major < 2 || (major == 2 && minor < 32) )); then
    echo "AWS CLI >= 2.32.0 is required for aws login; found: $version" >&2
    return 1
  fi

  # Use a subshell to keep stale environment credentials out of browser login
  # without changing the caller's environment if login is cancelled or fails.
  (
    unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SECURITY_TOKEN AWS_DEFAULT_PROFILE
    aws login --profile cs182-login --region us-east-1 "$@"
  ) || return 1

  # The pinned AWS provider and boto3 use this process-based compatibility profile.
  aws configure set credential_process \
    'aws configure export-credentials --profile cs182-login --format process' \
    --profile cs182-tools || return 1
  aws configure set region us-east-1 --profile cs182-tools || return 1

  unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SECURITY_TOKEN AWS_DEFAULT_PROFILE
  export AWS_PROFILE=cs182-tools
  export AWS_DEFAULT_REGION=us-east-1
  export AWS_REGION=us-east-1
  aws sts get-caller-identity --profile cs182-tools || return 1
  echo 'AWS_PROFILE=cs182-tools is active in this terminal.'
}

# Never exit the calling shell or change its errexit/nounset/pipefail settings.
if _cs182_aws_login "$@"; then
  unset -f _cs182_aws_login
else
  unset -f _cs182_aws_login
  return 1
fi
