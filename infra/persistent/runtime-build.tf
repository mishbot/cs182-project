variable "runtime_build_source_url" {
  description = "Public GitHub repository containing the runtime Dockerfile and buildspec."
  type        = string
  default     = "https://github.com/mishbot/cs182-project.git"
}

variable "runtime_build_compute_type" {
  type    = string
  default = "BUILD_GENERAL1_MEDIUM"
  validation {
    condition     = contains(["BUILD_GENERAL1_SMALL", "BUILD_GENERAL1_MEDIUM", "BUILD_GENERAL1_LARGE"], var.runtime_build_compute_type)
    error_message = "Use a supported small, medium, or large on-demand build container."
  }
}

variable "runtime_build_timeout" {
  description = "Maximum build duration in minutes."
  type        = number
  default     = 60
  validation {
    condition     = var.runtime_build_timeout >= 5 && var.runtime_build_timeout <= 480 && floor(var.runtime_build_timeout) == var.runtime_build_timeout
    error_message = "The build timeout must be an integer from 5 through 480 minutes."
  }
}

data "aws_partition" "current" {}

locals {
  runtime_build_name    = "${var.project}-runtime-build"
  runtime_build_arn     = "arn:${data.aws_partition.current.partition}:codebuild:${var.region}:${data.aws_caller_identity.current.account_id}:project/${local.runtime_build_name}"
  runtime_build_logs    = "/aws/codebuild/${local.runtime_build_name}"
  runtime_build_log_arn = "arn:${data.aws_partition.current.partition}:logs:${var.region}:${data.aws_caller_identity.current.account_id}:log-group:${local.runtime_build_logs}"
  logs_service          = "logs.${var.region}.${data.aws_partition.current.dns_suffix}"
}

resource "aws_ecr_repository" "runtime_build_cache" {
  name                 = "${var.project}-runtime-build-cache"
  image_tag_mutability = "MUTABLE"
  force_delete         = false
  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "runtime_build_cache" {
  repository = aws_ecr_repository.runtime_build_cache.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Expire superseded untagged BuildKit cache manifests after seven days"
      selection = {
        tagStatus   = "untagged"
        countType   = "sinceImagePushed"
        countUnit   = "days"
        countNumber = 7
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_kms_key" "runtime_build_logs" {
  description             = "Encrypt CloudWatch logs for ${local.runtime_build_name}"
  enable_key_rotation     = true
  deletion_window_in_days = 30
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "AccountAdministration"
        Effect    = "Allow"
        Principal = { AWS = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:root" }
        Action    = "kms:*"
        Resource  = "*"
      },
      {
        Sid       = "EncryptOnlyThisLogGroup"
        Effect    = "Allow"
        Principal = { Service = local.logs_service }
        Action    = ["kms:Encrypt", "kms:Decrypt", "kms:ReEncrypt*", "kms:GenerateDataKey*", "kms:DescribeKey"]
        Resource  = "*"
        Condition = { ArnEquals = { "kms:EncryptionContext:aws:logs:arn" = local.runtime_build_log_arn } }
      }
    ]
  })
}

resource "aws_kms_alias" "runtime_build_logs" {
  name          = "alias/${local.runtime_build_name}-logs"
  target_key_id = aws_kms_key.runtime_build_logs.key_id
}

resource "aws_cloudwatch_log_group" "runtime_build" {
  name              = local.runtime_build_logs
  retention_in_days = 14
  kms_key_id        = aws_kms_key.runtime_build_logs.arn
}

resource "aws_iam_role" "runtime_build" {
  name = local.runtime_build_name
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "codebuild.amazonaws.com" }
      Action    = "sts:AssumeRole"
      Condition = {
        StringEquals = { "aws:SourceAccount" = data.aws_caller_identity.current.account_id }
        ArnEquals    = { "aws:SourceArn" = local.runtime_build_arn }
      }
    }]
  })
}

# The buildspec is shell code, so these service-role actions follow the ECR
# remote-cache CodeBuild example in AWS's documentation. Launcher SDK operations
# have a separate operator permission set documented in infra/codebuild/README.md.
resource "aws_iam_role_policy" "runtime_build" {
  name = "runtime-build-and-cache"
  role = aws_iam_role.runtime_build.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = "*" },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer",
          "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload", "ecr:PutImage"
        ]
        Resource = [aws_ecr_repository.runtime.arn, aws_ecr_repository.runtime_build_cache.arn]
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "${local.runtime_build_log_arn}:*"
      }
    ]
  })
}

resource "aws_codebuild_project" "runtime" {
  name                   = local.runtime_build_name
  description            = "Build and test an exact public GitHub commit, then publish the runtime image"
  service_role           = aws_iam_role.runtime_build.arn
  build_timeout          = var.runtime_build_timeout
  queued_timeout         = 30
  concurrent_build_limit = 1

  artifacts {
    type = "NO_ARTIFACTS"
  }
  source {
    type                = "GITHUB"
    location            = var.runtime_build_source_url
    git_clone_depth     = 1
    report_build_status = false
    buildspec           = "infra/codebuild/buildspec.yml"
  }
  environment {
    type                        = "LINUX_CONTAINER"
    image                       = "aws/codebuild/standard:8.0"
    compute_type                = var.runtime_build_compute_type
    privileged_mode             = true
    image_pull_credentials_type = "CODEBUILD"

    environment_variable {
      name  = "RUNTIME_REPOSITORY_URI"
      value = aws_ecr_repository.runtime.repository_url
    }
    environment_variable {
      name  = "CACHE_REPOSITORY_URI"
      value = aws_ecr_repository.runtime_build_cache.repository_url
    }
    environment_variable {
      name  = "SOURCE_REPOSITORY_URL"
      value = var.runtime_build_source_url
    }
  }
  logs_config {
    cloudwatch_logs {
      status      = "ENABLED"
      group_name  = aws_cloudwatch_log_group.runtime_build.name
      stream_name = "runtime"
    }
  }

  depends_on = [aws_iam_role_policy.runtime_build]
}

output "runtime_build_project_name" { value = aws_codebuild_project.runtime.name }
output "runtime_build_cache_repository_url" { value = aws_ecr_repository.runtime_build_cache.repository_url }
