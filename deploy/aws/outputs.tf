output "api_url" {
  description = "Public HTTPS URL of the DELPHI API."
  value       = "https://${aws_apprunner_service.this.service_url}"
}

output "ecr_repository_url" {
  description = "Push the image here before creating the App Runner service."
  value       = aws_ecr_repository.app.repository_url
}

output "apprunner_service_arn" {
  description = "App Runner service ARN (for start-deployment on image updates)."
  value       = aws_apprunner_service.this.arn
}

output "rds_endpoint" {
  description = "RDS Postgres endpoint (private)."
  value       = aws_db_instance.this.address
}

output "snapshots_bucket" {
  description = "S3 bucket for the evidence snapshot lake."
  value       = aws_s3_bucket.snapshots.bucket
}

output "api_token" {
  description = "Bearer token clients must send (also stored in Secrets Manager)."
  value       = local.api_token
  sensitive   = true
}
