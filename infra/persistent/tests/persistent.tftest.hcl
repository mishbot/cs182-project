mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = { account_id = "123456789012" }
  }
  mock_data "aws_partition" {
    defaults = { partition = "aws", dns_suffix = "amazonaws.com" }
  }
  mock_resource "aws_iam_role" {
    defaults = { arn = "arn:aws:iam::123456789012:role/cs182-cddr-runtime-build" }
  }
  mock_resource "aws_kms_key" {
    defaults = {
      arn    = "arn:aws:kms:us-east-1:123456789012:key/11111111-1111-1111-1111-111111111111"
      key_id = "11111111-1111-1111-1111-111111111111"
    }
  }
}

run "runtime_cloud_build" {
  command = plan

  assert {
    condition     = aws_codebuild_project.runtime.name == "cs182-cddr-runtime-build" && aws_codebuild_project.runtime.source[0].type == "GITHUB" && aws_codebuild_project.runtime.source[0].location == "https://github.com/mishbot/cs182-project.git" && aws_codebuild_project.runtime.source[0].buildspec == "infra/codebuild/buildspec.yml"
    error_message = "CodeBuild must use the public GitHub repository and versioned buildspec."
  }
  assert {
    condition     = aws_codebuild_project.runtime.environment[0].type == "LINUX_CONTAINER" && aws_codebuild_project.runtime.environment[0].privileged_mode && aws_codebuild_project.runtime.environment[0].image == "aws/codebuild/standard:8.0" && aws_codebuild_project.runtime.environment[0].compute_type == "BUILD_GENERAL1_MEDIUM"
    error_message = "Build Docker natively on a privileged managed amd64 build container."
  }
  assert {
    condition     = length(aws_codebuild_project.runtime.vpc_config) == 0 && aws_codebuild_project.runtime.concurrent_build_limit == 1 && aws_codebuild_project.runtime.build_timeout == 60 && aws_codebuild_project.runtime.artifacts[0].type == "NO_ARTIFACTS"
    error_message = "Use on-demand managed networking, one build at a time, and no unused artifacts."
  }
  assert {
    condition     = aws_ecr_repository.runtime.image_tag_mutability == "IMMUTABLE" && aws_ecr_repository.runtime_build_cache.image_tag_mutability == "MUTABLE" && aws_ecr_repository.runtime.name != aws_ecr_repository.runtime_build_cache.name
    error_message = "Cache updates must never make runtime image tags mutable."
  }
  assert {
    condition     = jsondecode(aws_ecr_lifecycle_policy.runtime_build_cache.policy).rules[0].selection.tagStatus == "untagged" && jsondecode(aws_ecr_lifecycle_policy.runtime_build_cache.policy).rules[0].selection.countNumber == 7
    error_message = "Expire superseded untagged cache manifests after seven days."
  }
  assert {
    condition     = aws_cloudwatch_log_group.runtime_build.retention_in_days == 14 && aws_kms_key.runtime_build_logs.enable_key_rotation && jsondecode(aws_kms_key.runtime_build_logs.policy).Statement[1].Condition.ArnEquals["kms:EncryptionContext:aws:logs:arn"] == "arn:aws:logs:us-east-1:123456789012:log-group:/aws/codebuild/cs182-cddr-runtime-build"
    error_message = "Retain encrypted build logs for 14 days using a rotating key."
  }
  assert {
    condition     = jsondecode(aws_iam_role.runtime_build.assume_role_policy).Statement[0].Condition.StringEquals["aws:SourceAccount"] == "123456789012" && jsondecode(aws_iam_role.runtime_build.assume_role_policy).Statement[0].Condition.ArnEquals["aws:SourceArn"] == "arn:aws:codebuild:us-east-1:123456789012:project/cs182-cddr-runtime-build"
    error_message = "Only this account's build project may assume the build role."
  }
  assert {
    condition     = toset([for item in aws_codebuild_project.runtime.environment[0].environment_variable : item.name]) == toset(["RUNTIME_REPOSITORY_URI", "CACHE_REPOSITORY_URI", "SOURCE_REPOSITORY_URL"])
    error_message = "Wire the runtime and cache repositories into the build environment."
  }
  assert {
    condition     = output.runtime_build_project_name == aws_codebuild_project.runtime.name
    error_message = "Export the cloud build project name."
  }
}

variables {
  budget_email = "budget@example.com"
}

run "persistent_resources" {
  command = plan
  assert {
    condition     = aws_s3_bucket.storage["training-stats"].bucket == "cs182-cddr-123456789012-us-east-1-training-stats" && output.training_stats_bucket == aws_s3_bucket.storage["training-stats"].id && !aws_s3_bucket.storage["training-stats"].force_destroy && output.dev_server_inputs.training_stats_bucket == aws_s3_bucket.storage["training-stats"].id
    error_message = "Export a dedicated training-statistics bucket without forced destruction."
  }
  assert {
    condition     = aws_s3_bucket.storage["analyses"].bucket == "cs182-cddr-123456789012-us-east-1-analyses" && output.analysis_bucket == aws_s3_bucket.storage["analyses"].id && !aws_s3_bucket.storage["analyses"].force_destroy
    error_message = "Add the dedicated analyses bucket without permitting forced destruction."
  }

  assert {
    condition     = contains(keys(aws_s3_bucket.storage), "datasets") && aws_s3_bucket_versioning.datasets.versioning_configuration[0].status == "Enabled"
    error_message = "A dedicated versioned datasets bucket must exist."
  }
  assert {
    condition     = aws_ebs_volume.data.encrypted && aws_ebs_volume.data.size == 50 && aws_ebs_volume.data.type == "gp3"
    error_message = "The persistent data volume must be 50 GiB, encrypted gp3."
  }
  assert {
    condition     = aws_s3_bucket_versioning.models.versioning_configuration[0].status == "Enabled"
    error_message = "Model versioning must remain enabled."
  }
  assert {
    condition     = alltrue([for block in aws_s3_bucket_public_access_block.storage : block.block_public_acls && block.block_public_policy && block.ignore_public_acls && block.restrict_public_buckets])
    error_message = "All storage buckets must block all public access."
  }
  assert {
    condition     = aws_ecr_repository.runtime.image_tag_mutability == "IMMUTABLE"
    error_message = "Published image tags must be immutable."
  }
  assert {
    condition     = alltrue([for name in ["activation_bucket", "dataset_bucket", "analysis_bucket"] : contains(keys(output.dev_server_inputs), name)])
    error_message = "Export the activation and dataset buckets to the dev server."
  }

  assert {
    condition     = aws_budgets_budget.project.budget_type == "COST" && aws_budgets_budget.project.time_unit == "MONTHLY" && tonumber(aws_budgets_budget.project.limit_amount) == 50 && aws_budgets_budget.project.limit_unit == "USD"
    error_message = "The project must have a $50 monthly cost budget."
  }
  assert {
    condition     = one(aws_budgets_budget.project.cost_filter).name == "TagKeyValue" && one(aws_budgets_budget.project.cost_filter).values == tolist([format("user:Project$%s", var.project)])
    error_message = "The budget must track the configured Project tag."
  }
  assert {
    condition     = one(aws_budgets_budget.project.notification).threshold == 50 && one(aws_budgets_budget.project.notification).threshold_type == "PERCENTAGE" && one(aws_budgets_budget.project.notification).comparison_operator == "GREATER_THAN" && one(aws_budgets_budget.project.notification).notification_type == "ACTUAL" && one(aws_budgets_budget.project.notification).subscriber_email_addresses == toset([var.budget_email])
    error_message = "Email the configured recipient when actual spending exceeds 50% ($25)."
  }
}
