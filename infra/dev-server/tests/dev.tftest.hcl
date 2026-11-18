mock_provider "aws" {
  mock_data "aws_ebs_volume" {
    defaults = { availability_zone = "us-east-1a" }
  }
  mock_data "aws_s3_bucket" {
    defaults = { region = "us-east-1", arn = "arn:aws:s3:::fixture-models" }
  }
  override_data {
    target = data.aws_s3_bucket.datasets
    values = { region = "us-east-1", arn = "arn:aws:s3:::fixture-datasets" }
  }
  override_data {
    target = data.aws_s3_bucket.activations
    values = { region = "us-east-1", arn = "arn:aws:s3:::fixture-activations" }
  }
  mock_data "aws_ecr_repository" {
    defaults = { arn = "arn:aws:ecr:us-east-1:123456789012:repository/fixture-runtime" }
  }
}

variables {
  availability_zone = "us-east-1a"
  project           = "fixture-qwen"
  model_bucket      = "fixture-models"
  dataset_bucket    = "fixture-datasets"
  activation_bucket = "fixture-activations"
  ecr_repository    = "fixture-runtime"
  data_volume_id    = "vol-0123456789abcdef0"
  ssh_public_key    = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFixtureKeyOnlyForOfflineTests"
  ssh_cidr          = "203.0.113.10/32"
  ami_id            = "ami-0123456789abcdef0"
}

run "dev_resources" {
  command = plan

  assert {
    condition     = toset(jsondecode(aws_iam_role_policy.read.policy).Statement[1].Resource) == toset(["arn:aws:s3:::fixture-models/*", "arn:aws:s3:::fixture-datasets/*"]) && toset(jsondecode(aws_iam_role_policy.read.policy).Statement[1].Action) == toset(["s3:GetObject"])
    error_message = "The server must have read-only object access to the model and dataset buckets."
  }
  assert {
    condition     = aws_instance.dev.instance_type == "g4dn.xlarge" && aws_instance.dev.associate_public_ip_address
    error_message = "Default server must be a public g4dn.xlarge."
  }
  assert {
    condition     = aws_instance.dev.metadata_options[0].http_tokens == "required" && aws_instance.dev.metadata_options[0].http_put_response_hop_limit == 2
    error_message = "Require IMDSv2 and allow bridge containers to reach it."
  }
  assert {
    condition     = aws_instance.dev.root_block_device[0].encrypted && aws_instance.dev.root_block_device[0].delete_on_termination
    error_message = "Root disk must be encrypted and disposable."
  }
  assert {
    condition     = aws_volume_attachment.data.stop_instance_before_detaching
    error_message = "Stop the instance before detaching its persistent data disk."
  }
  assert {
    condition     = length(aws_security_group.ssh.ingress) == 1 && one(aws_security_group.ssh.ingress).from_port == 22 && one(aws_security_group.ssh.ingress).cidr_blocks == tolist([var.ssh_cidr])
    error_message = "Only the configured SSH source may reach the server."
  }
  assert {
    condition     = jsondecode(aws_iam_role_policy.activation_write.policy).Statement[0].Resource == "arn:aws:s3:::fixture-activations/*" && toset(jsondecode(aws_iam_role_policy.activation_write.policy).Statement[0].Action) == toset(["s3:PutObject", "s3:AbortMultipartUpload"])
    error_message = "Activation uploads must be scoped to the activation bucket, with multipart abort permission."
  }
  assert {
    condition     = jsondecode(aws_iam_role_policy.activation_read.policy).Statement[0].Resource == "arn:aws:s3:::fixture-activations" && toset(jsondecode(aws_iam_role_policy.activation_read.policy).Statement[0].Action) == toset(["s3:ListBucket"]) && toset(jsondecode(aws_iam_role_policy.activation_read.policy).Statement[0].Condition.StringLike["s3:prefix"]) == toset(["runs", "runs/*"])
    error_message = "Activation discovery must be limited to the runs prefix."
  }
  assert {
    condition     = jsondecode(aws_iam_role_policy.activation_read.policy).Statement[1].Resource == "arn:aws:s3:::fixture-activations/runs/*" && toset(jsondecode(aws_iam_role_policy.activation_read.policy).Statement[1].Action) == toset(["s3:GetObject"]) && length(jsondecode(aws_iam_role_policy.activation_read.policy).Statement) == 2
    error_message = "Analysis reads must be limited to activation run objects."
  }
}

run "reject_wrong_az" {
  command = plan
  variables {
    availability_zone = "us-east-1b"
  }
  expect_failures = [aws_instance.dev]
}

run "reject_public_ssh" {
  command = plan
  variables {
    ssh_cidr = "0.0.0.0/0"
  }
  expect_failures = [var.ssh_cidr]
}

run "reject_dataset_in_wrong_region" {
  command = plan
  override_data {
    target = data.aws_s3_bucket.datasets
    values = { region = "us-west-2", arn = "arn:aws:s3:::fixture-datasets" }
  }
  expect_failures = [aws_instance.dev]
}
