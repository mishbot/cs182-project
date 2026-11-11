mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = { account_id = "123456789012" }
  }
}

variables {
  budget_email = "budget@example.com"
}

run "persistent_resources" {
  command = plan

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
    error_message = "Both buckets must block all public access."
  }
  assert {
    condition     = aws_ecr_repository.runtime.image_tag_mutability == "IMMUTABLE"
    error_message = "Published image tags must be immutable."
  }
  assert {
    condition     = contains(keys(output.dev_server_inputs), "activation_bucket")
    error_message = "Export the activation bucket to the dev server."
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
