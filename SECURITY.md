# Security Policy

## Reporting a vulnerability

Please do **not** open a public issue for security vulnerabilities.

Use GitHub's [private vulnerability reporting](../../security/advisories/new) for this
repository. You will get an acknowledgement within a few days, and a fix or mitigation
plan before any public disclosure.

## Scope notes

- DELPHI's published API is designed to run behind your own authentication
  (`api_token` in the deploy configuration). Deployments without a token are not a
  supported configuration.
- Never commit real credentials: `.env` and `*.tfvars` are gitignored by design, and
  `gitleaks` runs in pre-commit. If you believe a secret has been committed, report it
  privately as above.
