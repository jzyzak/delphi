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

The intake surfaces let a client (e.g. a dashboard) type and formalize a
question cheaply before committing to a full forecast. Both take the same
body shape as forecast, but `as_of` is optional (it only enables the
"already resolved" refusal check) and nothing is written to the registry:

```bash
curl -s "$API/v1/classify" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question": "Will SpaceX reach orbit with Starship before 2027?"}'
# -> {"object": "question.classification", "classification": {"question_type": "binary", ...}}

curl -s "$API/v1/formalize" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question": "Will SpaceX reach orbit with Starship before 2027?"}'
# -> {"object": "question.formalization", "refused": false,
#     "formalized": {"text": ..., "resolution_criteria": ..., "close_time": ...}, ...}
```

A question that cannot be formalized (unresolvable, underspecified, opinion)
returns `200` with `refused: true` and a `refusal_reason` — refusal is a
product answer (CLAUDE.md §10), not an HTTP error.

## Long forecasts: the async job API (avoid App Runner's 120s cap)

App Runner enforces a **hard 120-second total request timeout** (not
configurable; client-side timeouts cannot override it), and a real forecast
usually runs longer — so the synchronous `POST /v1/forecast` gets killed with
a 504 mid-forecast. Long-running clients (e.g. a dashboard) must use the
**async job surface** instead:

```bash
# 1. Submit: returns 202 + a job id immediately. The idempotency key makes
#    retries safe — one key maps to exactly one job (no duplicate spend);
#    resubmitting a key returns the existing job (200) whatever its status.
JOB=$(curl -s "$API/v1/forecast/jobs" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{\"question\": \"Will SpaceX reach orbit with Starship before 2027?\", \
       \"as_of\": \"$NOW\", \"tier\": \"delphi\", \
       \"idempotency_key\": \"dash-req-42\"}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')

# 2. Poll with a long-poll wait (seconds, clamped to 90) until terminal:
curl -s "$API/v1/forecast/jobs/$JOB?wait=60" -H "Authorization: Bearer $TOKEN"
# -> {"object": "forecast.job", "status": "queued|running|succeeded|failed",
#     "result": <the full forecast response once succeeded>, "error": ...}
```

Client loop: submit once with an idempotency key, then `GET ...?wait=60` in a
loop until `status` is `succeeded` or `failed`, and read the forecast from
`result` (same shape as the synchronous response). **Use the `wait` long-poll
rather than rapid short GETs**: App Runner throttles instance CPU to ~0.01
vCPU when no request is in flight, so the open poll is also what keeps the
forecast worker running at full speed.

Operational notes:

- Jobs persist in Postgres (`DELPHI_PG_DSN`), so polls can land on any
  worker/instance. Without a DSN the store is in-memory and single-process
  (local `delphi serve` only).
- `DELPHI_JOB_WORKERS` (default 2) caps concurrent forecasts per API process;
  `DELPHI_JOB_TIMEOUT_S` (default 1800) is the stale timeout after which a
  running job whose worker died (deploy/crash) is reported `failed` instead of
  spinning forever. Queued jobs orphaned by a restart are revived by the next
  poll.
- The container runs gunicorn with `--threads` (`GUNICORN_THREADS`, default 8)
  so long-polls cannot starve the worker pool.
- A refused question is a **succeeded** job whose `result.delphi.refused` is
  `true` — refusal stays a product answer, not a job failure.

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
