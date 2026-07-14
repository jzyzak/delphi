variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "name_prefix" {
  description = "Prefix for all resource names."
  type        = string
  default     = "delphi"
}

# --- Secrets (sensitive). Prefer a gitignored *.tfvars or -var over committing. ---

variable "anthropic_api_key" {
  description = "Claude Console API key (stored in Secrets Manager)."
  type        = string
  sensitive   = true
}

variable "tavily_api_key" {
  description = "Tavily search API key (stored in Secrets Manager)."
  type        = string
  sensitive   = true
}

variable "api_token" {
  description = "Bearer token clients must present. Empty => a strong one is generated."
  type        = string
  sensitive   = true
  default     = ""
}

# --- Model tiers (optional overrides; defaults live in common/settings.py). ---

variable "llm_provider" {
  description = "LLM transport: 'anthropic' (direct API) or 'bedrock' (in-VPC)."
  type        = string
  default     = "anthropic"
}

variable "model_overrides" {
  description = "Optional DELPHI_MODEL_* overrides, e.g. {DELPHI_MODEL_OPUS=\"...\"}."
  type        = map(string)
  default     = {}
}

# --- Container image ---

variable "image_tag" {
  description = "Image tag to deploy from the created ECR repository."
  type        = string
  default     = "latest"
}

# --- App Runner sizing ---

variable "apprunner_cpu" {
  description = "App Runner vCPU (e.g. '1024' = 1 vCPU)."
  type        = string
  default     = "1024"
}

variable "apprunner_memory" {
  description = "App Runner memory in MB (e.g. '2048')."
  type        = string
  default     = "2048"
}

variable "web_concurrency" {
  description = "gunicorn worker count."
  type        = string
  default     = "2"
}

# --- RDS sizing ---

variable "db_instance_class" {
  description = "RDS instance class."
  type        = string
  default     = "db.t4g.micro"
}

variable "db_allocated_storage" {
  description = "RDS storage (GiB)."
  type        = number
  default     = 20
}

variable "db_name" {
  description = "Initial database name."
  type        = string
  default     = "delphi"
}

variable "db_username" {
  description = "RDS master username."
  type        = string
  default     = "delphi"
}

# --- Networking. Defaults to the account's default VPC + its subnets. ---

variable "vpc_id" {
  description = "VPC id for RDS + the App Runner VPC connector. Empty => default VPC."
  type        = string
  default     = ""
}

variable "subnet_ids" {
  description = "Subnet ids for RDS + the connector. Empty => default VPC subnets."
  type        = list(string)
  default     = []
}

variable "apprunner_unsupported_az_ids" {
  description = "AZ ids where App Runner is unavailable; excluded from the VPC connector."
  type        = list(string)
  default     = ["use1-az3"]
}

variable "apprunner_egress_mode" {
  description = "App Runner egress: 'VPC' (reach private RDS via connector + NAT) or 'DEFAULT' (public)."
  type        = string
  default     = "VPC"
}

variable "apprunner_start_command" {
  description = "Override the container start command (debug only). Empty = image CMD."
  type        = string
  default     = ""
}

variable "apprunner_inject_secrets" {
  description = "Inject Secrets Manager values as env vars (debug toggle)."
  type        = bool
  default     = true
}
