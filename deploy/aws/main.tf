# DELPHI on AWS: App Runner (container) + RDS Postgres (pgvector) + Secrets
# Manager + S3, wired for the direct Anthropic API (Bedrock optional).
#
# NOTE: this is a reviewed baseline; it was not `terraform apply`-validated in
# the authoring environment. Run `terraform init && terraform plan` and review
# before applying. App Runner requires the image to exist in ECR first, so
# deploy in two steps (see deploy/README.md):
#   terraform apply -target=aws_ecr_repository.app   # 1. create the repo
#   <build + push the image>                          # 2. push :image_tag
#   terraform apply                                   # 3. create everything else

data "aws_caller_identity" "current" {}

data "aws_vpc" "default" {
  count   = var.vpc_id == "" ? 1 : 0
  default = true
}

data "aws_subnets" "default" {
  count = length(var.subnet_ids) == 0 ? 1 : 0
  filter {
    name   = "vpc-id"
    values = [var.vpc_id == "" ? data.aws_vpc.default[0].id : var.vpc_id]
  }
}

# App Runner is not available in every AZ (e.g. use1-az3 in us-east-1). Look up
# each candidate subnet's AZ so we can exclude unsupported ones from the
# connector (RDS, by contrast, is fine in any AZ).
data "aws_subnet" "selected" {
  for_each = toset(local.subnet_ids)
  id       = each.value
}

data "aws_vpc" "selected" {
  id = local.vpc_id
}

resource "random_password" "db" {
  length  = 32
  special = false
}

resource "random_password" "api_token" {
  count   = var.api_token == "" ? 1 : 0
  length  = 40
  special = false
}

locals {
  vpc_id     = var.vpc_id == "" ? data.aws_vpc.default[0].id : var.vpc_id
  subnet_ids = length(var.subnet_ids) == 0 ? data.aws_subnets.default[0].ids : var.subnet_ids

  # Subnets in App Runner-supported AZs only (filtered by AZ id, which is stable
  # across accounts, unlike AZ names).
  apprunner_subnet_ids = sort([
    for id, s in data.aws_subnet.selected : id
    if !contains(var.apprunner_unsupported_az_ids, s.availability_zone_id)
  ])

  # Supported AZ names (for placing the private subnets) + one public subnet to
  # host the NAT gateway.
  supported_azs = sort(distinct([
    for id, s in data.aws_subnet.selected : s.availability_zone
    if !contains(var.apprunner_unsupported_az_ids, s.availability_zone_id)
  ]))
  nat_public_subnet_id = local.apprunner_subnet_ids[0]

  api_token = var.api_token == "" ? random_password.api_token[0].result : var.api_token

  db_dsn = format(
    "postgresql://%s:%s@%s:5432/%s",
    var.db_username,
    random_password.db.result,
    aws_db_instance.this.address,
    var.db_name,
  )
}

# --- Container registry -------------------------------------------------------

resource "aws_ecr_repository" "app" {
  name                 = "${var.name_prefix}-api"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# --- Evidence snapshot lake ---------------------------------------------------

resource "aws_s3_bucket" "snapshots" {
  bucket = "${var.name_prefix}-snapshots-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_public_access_block" "snapshots" {
  bucket                  = aws_s3_bucket.snapshots.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- Secrets ------------------------------------------------------------------

resource "aws_secretsmanager_secret" "anthropic" {
  name = "${var.name_prefix}/anthropic-api-key"
}

resource "aws_secretsmanager_secret_version" "anthropic" {
  secret_id     = aws_secretsmanager_secret.anthropic.id
  secret_string = var.anthropic_api_key
}

resource "aws_secretsmanager_secret" "tavily" {
  name = "${var.name_prefix}/tavily-api-key"
}

resource "aws_secretsmanager_secret_version" "tavily" {
  secret_id     = aws_secretsmanager_secret.tavily.id
  secret_string = var.tavily_api_key
}

resource "aws_secretsmanager_secret" "api_token" {
  name = "${var.name_prefix}/api-token"
}

resource "aws_secretsmanager_secret_version" "api_token" {
  secret_id     = aws_secretsmanager_secret.api_token.id
  secret_string = local.api_token
}

resource "aws_secretsmanager_secret" "db_dsn" {
  name = "${var.name_prefix}/pg-dsn"
}

resource "aws_secretsmanager_secret_version" "db_dsn" {
  secret_id     = aws_secretsmanager_secret.db_dsn.id
  secret_string = local.db_dsn
}

# --- Networking: RDS reachable only from the App Runner connector -------------

resource "aws_security_group" "apprunner" {
  name        = "${var.name_prefix}-apprunner"
  description = "App Runner VPC connector egress"
  vpc_id      = local.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "rds" {
  name        = "${var.name_prefix}-rds"
  description = "Postgres access from App Runner only"
  vpc_id      = local.vpc_id

  ingress {
    description     = "Postgres from App Runner"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.apprunner.id]
  }
}

resource "aws_db_subnet_group" "this" {
  name       = "${var.name_prefix}-db"
  subnet_ids = local.subnet_ids
}

resource "aws_db_instance" "this" {
  identifier             = "${var.name_prefix}-pg"
  engine                 = "postgres"
  engine_version         = "16"
  instance_class         = var.db_instance_class
  allocated_storage      = var.db_allocated_storage
  db_name                = var.db_name
  username               = var.db_username
  password               = random_password.db.result
  db_subnet_group_name   = aws_db_subnet_group.this.name
  vpc_security_group_ids  = [aws_security_group.rds.id]
  publicly_accessible    = false
  storage_encrypted      = true
  skip_final_snapshot    = true
  deletion_protection    = false
  apply_immediately      = true
}

# --- IAM: instance role (reads secrets/S3) + access role (pulls ECR) ----------

data "aws_iam_policy_document" "instance_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["tasks.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "instance" {
  name               = "${var.name_prefix}-apprunner-instance"
  assume_role_policy = data.aws_iam_policy_document.instance_assume.json
}

data "aws_iam_policy_document" "instance" {
  statement {
    sid     = "ReadSecrets"
    actions = ["secretsmanager:GetSecretValue"]
    resources = [
      aws_secretsmanager_secret.anthropic.arn,
      aws_secretsmanager_secret.tavily.arn,
      aws_secretsmanager_secret.api_token.arn,
      aws_secretsmanager_secret.db_dsn.arn,
    ]
  }

  # App Runner decrypts the secret values with KMS; without this the container
  # fails to start (CREATE_FAILED) before it can emit any logs. Scoped to the
  # Secrets Manager service via a condition (covers the default aws/secretsmanager key).
  statement {
    sid       = "DecryptSecrets"
    actions   = ["kms:Decrypt"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["secretsmanager.${var.aws_region}.amazonaws.com"]
    }
  }

  statement {
    sid       = "Snapshots"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.snapshots.arn, "${aws_s3_bucket.snapshots.arn}/*"]
  }

  # Only used when var.llm_provider = "bedrock"; harmless otherwise.
  statement {
    sid       = "Bedrock"
    actions   = ["bedrock:InvokeModel"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "instance" {
  name   = "${var.name_prefix}-instance"
  role   = aws_iam_role.instance.id
  policy = data.aws_iam_policy_document.instance.json
}

data "aws_iam_policy_document" "access_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["build.apprunner.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "access" {
  name               = "${var.name_prefix}-apprunner-access"
  assume_role_policy = data.aws_iam_policy_document.access_assume.json
}

resource "aws_iam_role_policy_attachment" "access_ecr" {
  role       = aws_iam_role.access.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

# --- App Runner ---------------------------------------------------------------

# Private subnets for the App Runner connector: they reach RDS intra-VPC and
# egress to the internet (Claude/Tavily) via the NAT gateway. The default VPC's
# subnets are public and lack a NAT route, so App Runner containers placed there
# can't start; these dedicated private subnets fix that.
resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = local.vpc_id
  availability_zone = element(local.supported_azs, count.index)
  cidr_block        = cidrsubnet(data.aws_vpc.selected.cidr_block, 8, 200 + count.index)

  tags = { Name = "${var.name_prefix}-private-${count.index + 1}" }
}

resource "aws_eip" "nat" {
  domain = "vpc"
  tags   = { Name = "${var.name_prefix}-nat" }
}

resource "aws_nat_gateway" "this" {
  allocation_id = aws_eip.nat.id
  subnet_id     = local.nat_public_subnet_id
  tags          = { Name = "${var.name_prefix}-nat" }
}

resource "aws_route_table" "private" {
  vpc_id = local.vpc_id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.this.id
  }

  tags = { Name = "${var.name_prefix}-private" }
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

resource "aws_apprunner_vpc_connector" "this" {
  vpc_connector_name = "${var.name_prefix}-connector"
  subnets            = aws_subnet.private[*].id
  security_groups    = [aws_security_group.apprunner.id]
}

resource "aws_apprunner_service" "this" {
  service_name = "${var.name_prefix}-api"

  source_configuration {
    auto_deployments_enabled = false

    authentication_configuration {
      access_role_arn = aws_iam_role.access.arn
    }

    image_repository {
      image_identifier      = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
      image_repository_type = "ECR"

      image_configuration {
        port          = "8080"
        start_command = var.apprunner_start_command != "" ? var.apprunner_start_command : null

        runtime_environment_variables = merge(
          {
            DELPHI_LLM_PROVIDER = var.llm_provider
            WEB_CONCURRENCY     = var.web_concurrency
          },
          var.model_overrides,
        )

        runtime_environment_secrets = var.apprunner_inject_secrets ? {
          DELPHI_SECRET_ANTHROPIC_API_KEY = aws_secretsmanager_secret.anthropic.arn
          DELPHI_SECRET_TAVILY_API_KEY    = aws_secretsmanager_secret.tavily.arn
          DELPHI_SECRET_API_TOKEN         = aws_secretsmanager_secret.api_token.arn
          DELPHI_PG_DSN                   = aws_secretsmanager_secret.db_dsn.arn
        } : {}
      }
    }
  }

  instance_configuration {
    cpu               = var.apprunner_cpu
    memory            = var.apprunner_memory
    instance_role_arn = aws_iam_role.instance.arn
  }

  network_configuration {
    egress_configuration {
      egress_type       = var.apprunner_egress_mode
      vpc_connector_arn = var.apprunner_egress_mode == "VPC" ? aws_apprunner_vpc_connector.this.arn : null
    }
  }

  # TCP is the most robust default: it confirms gunicorn is listening without
  # depending on health-check HTTP semantics/timeouts.
  health_check_configuration {
    protocol            = "TCP"
    interval            = 10
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 5
  }
}
