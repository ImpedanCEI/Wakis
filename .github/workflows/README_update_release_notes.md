# Automated Release Notes Workflow

This document explains how the automated release notes generation workflow works.

## Overview

The `update_release_notes.yml` workflow automatically updates the `release.md` file whenever:
- Commits are pushed to the `main` branch
- Pull requests are opened, synchronized, or reopened targeting the `main` branch

## How It Works

### Commit Range Detection

- **For Pull Requests**: Analyzes commits between the PR base and head
- **For Push Events**: Analyzes commits since the last tag, or last 50 commits if no tags exist

### Categorization

The workflow categorizes commits based on conventional commit prefixes:

- **🚀 Features**: Commits starting with `feature:` or `feat:`
- **🧪 Tests**: Commits starting with `test:`
- **📚 Documentation**: Commits starting with `docs:`
- **⚙️ Build & Compatibility**: Commits starting with `build:` or `ci:`
- **🐛 Bugfixes**: Commits starting with `fix:` or `bugfix:`
- **🎨 Style**: Commits starting with `style:`

### Release Notes Structure

The workflow generates an "Unreleased" section at the top of `release.md` with:

1. **New Features**: List of feature commits
2. **Other Tag Highlights**: Organized by category (Tests, Docs, Build)
3. **Bugfixes**: List of bugfix commits
4. **Full Changelog**: Statistics table showing commit distribution and complete commit list

### Workflow Behavior

1. **On Push to Main**: The workflow commits and pushes the updated `release.md` directly to the main branch
2. **On Pull Request**: The workflow commits the updated `release.md` to the PR branch

The commit message includes `[skip ci]` for push events to prevent triggering the workflow recursively.

## Using Conventional Commit Messages

To take full advantage of automatic categorization, use conventional commit prefixes:

```bash
# Examples
git commit -m "feat: add new GPU acceleration support"
git commit -m "fix: resolve memory leak in solver"
git commit -m "docs: update installation guide"
git commit -m "test: add unit tests for geometry module"
git commit -m "style: format code with black"
git commit -m "build: update numpy dependency"
```

## Manual Release Process

When you're ready to publish a new release:

1. Review the "Unreleased" section in `release.md`
2. Manually edit it to:
   - Change "# Unreleased" to "# v{version}"
   - Add a descriptive summary of the release
   - Organize and expand the automatically generated content
   - Add any additional sections (e.g., "New Contributors")
3. Commit the changes
4. Tag the release using `release.sh` or manually

The next time the workflow runs, it will create a new "Unreleased" section above your versioned release.

## Disabling the Workflow

If you need to temporarily disable automatic release notes:

1. Rename the workflow file or move it to a different directory
2. Or add a condition to skip the workflow in certain cases

## Troubleshooting

- **No changes detected**: The workflow only commits if there are actual changes to `release.md`
- **Permission errors**: Ensure the workflow has `contents: write` permission
- **Commit range issues**: For repositories with shallow clones, increase fetch-depth in checkout action
