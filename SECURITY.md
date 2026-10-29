# Security policy

## Scope

This repository is a **simulation-only** reinforcement-learning project.
It neither controls physical robots nor supplies safety-rated motion planning.

## Reporting vulnerabilities

Please avoid disclosing exploitable details in public issues. Use GitHub's
private security-advisory reporting feature where available; otherwise
contact the repository maintainer through a private channel. Do not upload
credentials, production datasets, or confidential models to public issues.

## Artifact trust boundary

- MuJoCo environment XML, trained model ZIP archives, and Python pickle
  replay buffers must be treated as **untrusted inputs** when obtained
  from third parties. Stable-Baselines3 loading and pickle deserialization
  should be restricted to artifacts from trusted sources in an isolated
  runtime. SHA-256 checks here detect accidental corruption, **not**
  authenticity or malicious tampering.
- Never execute experiments with elevated privileges.
- Avoid mounting secrets or sensitive datasets into experimental runners.
- Rendered trajectories are demonstrations, not evidence of real-world
  safe control.
- GitHub Actions tests exercise simulation functionality only.

## Supported deployment

Python 3.11 with dependency ranges in `requirements.txt`.
The repository does not yet ship signed binary releases, a pinned multi-OS
lockfile, SBOM, or vulnerability attestation; those are release-blocking
tasks for deployments with stronger security requirements.
