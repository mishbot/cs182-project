terraform {
  required_version = ">= 1.8.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.100"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = var.project }
  }
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "availability_zone" {
  type    = string
  default = "us-east-1a"
}

variable "project" {
  type    = string
  default = "cs182-cddr"
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{2,29}$", var.project))
    error_message = "Use 3–30 lowercase letters, digits, or hyphens, beginning with a letter."
  }
}

variable "budget_email" {
  type        = string
  description = "Email address for the monthly project budget's $25 spending alert."
  validation {
    condition     = can(regex("^[^[:space:]@]+@[^[:space:]@]+[.][^[:space:]@]+$", var.budget_email))
    error_message = "Supply a valid recipient email address for budget alerts."
  }
}

data "aws_caller_identity" "current" {}
data "aws_availability_zone" "selected" {
  name = var.availability_zone
}

locals {
  prefix = "${var.project}-${data.aws_caller_identity.current.account_id}-${var.region}"
}

resource "aws_s3_bucket" "storage" {
  for_each      = toset(["models", "activations"])
  bucket        = "${local.prefix}-${each.key}"
  force_destroy = false
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_public_access_block" "storage" {
  for_each                = toset(["models", "activations"])
  bucket                  = aws_s3_bucket.storage[each.key].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "storage" {
  for_each = toset(["models", "activations"])
  bucket   = aws_s3_bucket.storage[each.key].id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "storage" {
  for_each = toset(["models", "activations"])
  bucket   = aws_s3_bucket.storage[each.key].id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.storage["models"].id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_policy" "tls" {
  for_each = toset(["models", "activations"])
  bucket   = aws_s3_bucket.storage[each.key].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "RequireTLS"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource  = [aws_s3_bucket.storage[each.key].arn, "${aws_s3_bucket.storage[each.key].arn}/*"]
      Condition = { Bool = { "aws:SecureTransport" = "false" } }
    }]
  })
}

resource "aws_ecr_repository" "runtime" {
  name                 = "${var.project}-runtime"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = false
  image_scanning_configuration {
    scan_on_push = true
  }
  encryption_configuration {
    encryption_type = "AES256"
  }
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_ebs_volume" "data" {
  availability_zone = data.aws_availability_zone.selected.name
  type              = "gp3"
  size              = 50
  encrypted         = true
  tags              = { Name = "${var.project}-data" }
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_budgets_budget" "project" {
  name         = "${var.project}-monthly"
  budget_type  = "COST"
  limit_amount = "50"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # Activate Project as a cost allocation tag in AWS Billing before relying on this filter.
  cost_filter {
    name   = "TagKeyValue"
    values = [format("user:Project$%s", var.project)]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 50
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }
}

output "dev_server_inputs" {
  description = "Export with tofu output -json dev_server_inputs to the dev-server .auto.tfvars.json file."
  value = {
    region            = var.region
    availability_zone = aws_ebs_volume.data.availability_zone
    project           = var.project
    model_bucket      = aws_s3_bucket.storage["models"].id
    activation_bucket = aws_s3_bucket.storage["activations"].id
    ecr_repository    = aws_ecr_repository.runtime.name
    data_volume_id    = aws_ebs_volume.data.id
  }
}

output "model_bucket" { value = aws_s3_bucket.storage["models"].id }
output "activation_bucket" { value = aws_s3_bucket.storage["activations"].id }
output "repository_url" { value = aws_ecr_repository.runtime.repository_url }
output "data_volume_id" { value = aws_ebs_volume.data.id }
