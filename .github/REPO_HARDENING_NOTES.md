# Repository Hardening Notes

This document describes repository protection settings that can be enabled in GitHub to enforce quality and security standards.

## Branch Protection Rules

To enable branch protection for the `main` branch:

1. Navigate to: Settings → Branches → Add branch protection rule
2. Branch name pattern: `main`
3. Enable the following settings:

### Required Checks
- ✅ Require status checks to pass before merging
  - ✅ Require branches to be up to date before merging
  - Required status checks:
    - `test` (from CI workflow)

### Pull Request Requirements
- ✅ Require a pull request before merging
  - ✅ Require approvals: 1
  - ✅ Dismiss stale pull request approvals when new commits are pushed
  - ✅ Require review from Code Owners (requires CODEOWNERS file)

### Additional Restrictions
- ✅ Require conversation resolution before merging
- ✅ Do not allow bypassing the above settings (recommended for production)

## Code Scanning

Enable GitHub Advanced Security features if available:

1. Navigate to: Settings → Code security and analysis
2. Enable:
   - Dependency graph (usually enabled by default)
   - Dependabot alerts
   - Dependabot security updates
   - Code scanning (CodeQL)

## Workflow Permissions

Ensure GitHub Actions has appropriate permissions:

1. Navigate to: Settings → Actions → General
2. Workflow permissions:
   - ✅ Read and write permissions (for publishing artifacts)
   - ✅ Allow GitHub Actions to create and approve pull requests (optional)

## Implementation Status

These settings are **not yet enforced** to allow initial development. Once the repository is stable and team is ready:

1. Enable branch protection on `main`
2. Require CI checks to pass
3. Require code owner review for sensitive paths
4. Enable Dependabot for security updates

## CI Checks

The CI workflow (`.github/workflows/ci.yml`) runs the following checks:

1. **Python compilation** - Ensures all Python modules are syntactically valid
2. **Unicode control character scan** - Detects potentially dangerous Unicode in source files
3. **Smoke test** - Runs full 3D pipeline to verify end-to-end functionality

All checks must pass before merging to `main` (once branch protection is enabled).

## References

- [GitHub Branch Protection](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [GitHub Code Owners](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- [GitHub Advanced Security](https://docs.github.com/en/get-started/learning-about-github/about-github-advanced-security)
