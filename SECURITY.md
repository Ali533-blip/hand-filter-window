# Security Policy

## Reporting a vulnerability

This is a hobby/portfolio project with no internet-facing services — it runs
locally, reads only your webcam, and writes files only to its own folder.
Still, if you find something that worries you:

Please **do not open a public issue** for security problems. Instead send a
private report to the maintainer via GitHub's
[private vulnerability reporting](https://docs.github.com/articles/managing-security-vulnerabilities)
(Repository → Security → Report a vulnerability), or open a draft PR with the
fix.

## What we take seriously

- Code execution from untrusted inputs (e.g. a model file swap).
- Any path/file-handling bug that could delete or overwrite files outside
  the app folder.
- Dependency issues flagged by `pip-audit` in `requirements.txt`.

## Supported versions

Security fixes are applied to the latest release on `main`. Older tags are
not maintained.