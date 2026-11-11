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
variable "availability_zone" { type = string }
variable "project" { type = string }
variable "model_bucket" { type = string }
variable "activation_bucket" { type = string }
variable "ecr_repository" { type = string }
variable "data_volume_id" { type = string }
variable "ssh_public_key" {
  type        = string
  description = "Contents of an SSH public key, never a private key."
  validation {
    condition     = can(regex("^(ssh-ed25519|ssh-rsa|ecdsa-sha2-[^ ]+) ", var.ssh_public_key))
    error_message = "Supply an OpenSSH public key."
  }
}
variable "ssh_cidr" {
  type        = string
  description = "Trusted public IPv4 CIDR allowed to SSH, normally your-ip/32."
  validation {
    condition     = can(cidrnetmask(var.ssh_cidr)) && !endswith(var.ssh_cidr, "/0")
    error_message = "Supply a valid restricted IPv4 CIDR; /0 is not allowed."
  }
}
variable "instance_type" {
  type    = string
  default = "g4dn.xlarge"
}
variable "ami_id" {
  type        = string
  default     = null
  description = "Optional pinned Ubuntu 22.04 x86_64 DLAMI ID with NVIDIA driver and AWS CLI."
}

data "aws_ssm_parameter" "dlami" {
  count = var.ami_id == null ? 1 : 0
  name  = "/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.5-ubuntu-22.04/latest/ami-id"
}
data "aws_ebs_volume" "data" {
  filter {
    name   = "volume-id"
    values = [var.data_volume_id]
  }
}
data "aws_s3_bucket" "models" { bucket = var.model_bucket }
data "aws_s3_bucket" "activations" { bucket = var.activation_bucket }
data "aws_ecr_repository" "runtime" { name = var.ecr_repository }

resource "aws_vpc" "main" {
  cidr_block           = "10.182.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
}
resource "aws_subnet" "public" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.182.1.0/24"
  availability_zone = var.availability_zone
}
resource "aws_internet_gateway" "main" { vpc_id = aws_vpc.main.id }
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
}
resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}
resource "aws_security_group" "ssh" {
  name_prefix = "${var.project}-ssh-"
  vpc_id      = aws_vpc.main.id
  ingress {
    description = "SSH from trusted network"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
resource "aws_key_pair" "ssh" {
  key_name_prefix = "${var.project}-"
  public_key      = var.ssh_public_key
}

resource "aws_iam_role" "dev" {
  name_prefix = "${var.project}-dev-"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow", Action = "sts:AssumeRole", Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}
resource "aws_iam_role_policy" "read" {
  role = aws_iam_role.dev.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:ListBucket"], Resource = data.aws_s3_bucket.models.arn },
      { Effect = "Allow", Action = ["s3:GetObject"], Resource = "${data.aws_s3_bucket.models.arn}/*" },
      { Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = "*" },
      {
        Effect   = "Allow"
        Action   = ["ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"]
        Resource = data.aws_ecr_repository.runtime.arn
      }
    ]
  })
}
resource "aws_iam_instance_profile" "dev" { role = aws_iam_role.dev.name }

resource "aws_iam_role_policy" "activation_write" {
  role = aws_iam_role.dev.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:PutObject", "s3:AbortMultipartUpload"]
      Resource = "${data.aws_s3_bucket.activations.arn}/*"
    }]
  })
}

resource "aws_instance" "dev" {
  ami                         = var.ami_id != null ? var.ami_id : data.aws_ssm_parameter.dlami[0].value
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.public.id
  associate_public_ip_address = true
  vpc_security_group_ids      = [aws_security_group.ssh.id]
  key_name                    = aws_key_pair.ssh.key_name
  iam_instance_profile        = aws_iam_instance_profile.dev.name
  user_data                   = templatefile("${path.module}/bootstrap.sh.tftpl", { volume_id = var.data_volume_id })
  user_data_replace_on_change = true
  root_block_device {
    volume_type           = "gp3"
    volume_size           = 100
    encrypted             = true
    delete_on_termination = true
  }
  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 2 # Allow Docker bridge containers to use instance-role credentials.
  }
  tags       = { Name = "${var.project}-dev" }
  depends_on = [aws_route_table_association.public, aws_iam_role_policy.read, aws_iam_role_policy.activation_write]
  lifecycle {
    precondition {
      condition     = data.aws_ebs_volume.data.availability_zone == var.availability_zone
      error_message = "The server and persistent EBS volume must share an availability zone."
    }
    precondition {
      condition     = data.aws_s3_bucket.models.region == var.region
      error_message = "The model bucket and dev server must share a region."
    }
    precondition {
      condition     = data.aws_s3_bucket.activations.region == var.region
      error_message = "The activation bucket and dev server must share a region."
    }
  }
}
resource "aws_volume_attachment" "data" {
  device_name                    = "/dev/sdf"
  volume_id                      = var.data_volume_id
  instance_id                    = aws_instance.dev.id
  stop_instance_before_detaching = true
}

output "instance_id" { value = aws_instance.dev.id }
output "public_ip" { value = aws_instance.dev.public_ip }
output "ssh_command" { value = "ssh ubuntu@${aws_instance.dev.public_ip}" }
output "ami_id" { value = aws_instance.dev.ami }
