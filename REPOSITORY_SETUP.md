# Repository maintenance

This repository is already public and initialized. Do not run `git init` again, fabricate earlier commits, or force-push `main` to rewrite the published history.

## Before each push

Run from the repository root:

```powershell
python -m ruff check .
python -m pytest
git diff --check
```

For native or build-system changes, also run `BUILD_NATIVE.ps1` on Windows and confirm the GitHub Actions matrix is green after the push.

## Commit discipline

- Keep each commit focused on one correctness, testing, documentation, or build concern.
- Use descriptive imperative messages.
- Inspect `git status` and `git diff` before staging.
- Stage explicit files instead of using `git add -f`.
- Never manufacture development history; design evolution is documented in `DECISIONS.md` and `CHANGELOG.md`.

## Do not commit

The `.gitignore` excludes local environments, native builds, recordings, generated reports, logs, compiled modules, and release archives. Do not override those exclusions.

Portable Windows packages belong in GitHub Releases, not in the source tree.
