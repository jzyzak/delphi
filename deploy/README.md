# Deploying the DELPHI API on AWS

This hosts the published API (`api.wsgi:application`) on **AWS App Runner** with
a **managed HTTPS URL**, backed by **RDS Postgres** (pgvector), **Secrets
Manager** (keys + the bearer token), and an **S3** evidence-snapshot bucket. The
same container image runs locally and in the cloud.

```
client ──HTTPS──▶ App Runner (gunicorn ▸ api.wsgi) ──VPC connector──▶ RDS Postgres (private)
                        │                        (private subnets)     Secrets Manager (keys, token, DSN)
                        └── egress ──▶ NAT gateway ──▶ Claude API + Tavily   S3 (snapshots)
```

The endpoint is **fail-closed on auth**: it will not serve forecasts unless
`DELPHI_SECRET_API_TOKEN` is set, and every forecast request must send
`Authorization: Bearer <token>`. Health/readiness stay open for probes.

> **Status:** this Terraform has been applied end-to-end and serves live
> forecasts. Still, review `terraform plan` before applying in your account.

## Prerequisites

- AWS account + credentials (`aws configure` / SSO), with rights to create the
  resources below (App Runner, RDS, VPC/NAT, Secrets Manager, S3, IAM, ECR).
- **Terraform ≥ 1.6.** Not in Homebrew core — install from HashiCorp's tap:
  `brew tap hashicorp/tap && brew install hashicorp/tap/terraform` (or use
  OpenTofu: `brew install opentofu`, substituting `tofu` for `terraform`).
- Docker (to build + push the image).
- A Claude Console API key and a Tavily API key.

## Step 1 — configure

```bash
cd deploy/aws
cp terraform.tfvars.example terraform.tfvars   # gitignored
# edit terraform.tfvars: anthropic_api_key, tavily_api_key, aws_region
terraform init
```

## Step 2 — create the ECR repo, then build + push the image

App Runner needs the image to exist before the service is created, so create the
repo first:

```bash
terraform apply -target=aws_ecr_repository.app
REPO=$(terraform output -raw ecr_repository_url)
REGION=$(terraform output -raw ecr_repository_url | cut -d. -f4)
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${REPO%/*}"
docker build --platform linux/amd64 --provenance=false -t "${REPO}:latest" ../..
docker push "${REPO}:latest"
```

Two non-obvious but required details (both learned the hard way):

- `--platform linux/amd64` — App Runner runs x86_64; build for it explicitly on
  Apple Silicon.
- `--provenance=false` — without it, `docker buildx` pushes a multi-manifest OCI
  *index* with an attestation entry, which App Runner pulls but **cannot run**
  (the service fails with `CREATE_FAILED` and no application logs). This flag
  forces a single-platform image manifest.
- Use `"${REPO}:latest"` with **braces**. In zsh, `"$REPO:latest"` triggers the
  `:l` (lowercase) modifier and mangles the tag to `...delphi-apiatest`.

## Step 3 — create everything else

```bash
terraform apply
terraform output api_url            # your HTTPS endpoint
terraform output -raw api_token     # the bearer token clients must send
```

App Runner pulls the image, attaches the VPC connector, and health-checks
`/healthz`. First deploy takes a few minutes.

## Step 4 — enable pgvector (one-time)

The stores auto-migrate their schema on first connect, which includes
`CREATE EXTENSION IF NOT EXISTS vector;`. The RDS master user created here has
permission to install it, so no manual step is normally needed. If migrations
report the extension is unavailable, connect once (e.g. via a bastion or a
temporary `publicly_accessible = true`) and run `CREATE EXTENSION vector;`.

## Step 5 — call it

```bash
API=$(terraform output -raw api_url)
TOKEN=$(terraform output -raw api_token)
NOW=$(python3 -c "import datetime; print(datetime.datetime.now(datetime.UTC).isoformat())")

curl -s "$API/v1/forecast" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"question\": \"Will SpaceX reach orbit with Starship before 2027?\", \"as_of\": \"$NOW\", \"tier\": \"delphi\"}" \
  | python3 -m json.tool
```

`tier` may be `delphi` (fixed pipeline) or `delphi_deep` (conductor).

## Updating the deployed version

```bash
docker build --platform linux/amd64 --provenance=false -t "${REPO}:latest" ../..
docker push "${REPO}:latest"
aws apprunner start-deployment --service-arn "$(terraform output -raw apprunner_service_arn)"
```

(Or bump `var.image_tag` to an immutable tag and `terraform apply`.)

## Notes & cost

- **Cost.** App Runner bills for provisioned + active container time; RDS and the
  **NAT gateway** bill hourly (the NAT is ~$32/mo + data — it gives the private
  container outbound access to the Claude/Tavily APIs); the big variable cost is
  **Claude API usage** — one `delphi` forecast fans out many model calls, and
  `delphi_deep` more. Start small.
- **Networking.** The container runs in dedicated **private subnets** (created by
  this stack) so it can reach the private RDS intra-VPC *and* the internet via the
  NAT gateway. RDS stays private; only the connector's security group reaches 5432.
  For a quick manual DB poke, temporarily set `publicly_accessible = true` + a
  home-IP ingress rule, then revert.
- **Debug toggles.** `apprunner_egress_mode`, `apprunner_inject_secrets`, and
  `apprunner_start_command` variables exist for troubleshooting; leave them at
  their defaults for a normal deploy.
- **Bedrock (optional).** Set `llm_provider = "bedrock"` + `model_overrides`
  with Bedrock-style ids and grant the instance role model access; egress to the
  Claude API is then unnecessary.
- **Teardown.** `terraform destroy` (the S3 bucket must be emptied first; ECR is
  `force_delete = true`).

## Alternative: ECS Fargate

App Runner is the least-ops path. If you need finer control (custom domains via
ALB, private networking, sidecars), the same image runs on **ECS Fargate behind
an ALB** with the identical env-var/secret contract; that variant isn't scripted
here.
