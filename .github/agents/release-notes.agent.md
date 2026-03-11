---
description: 'Generate release notes based on template'
tools: []
---

# Purpose
Generate release notes for each new release by summarizing all commits since the previous release, following the structure and style below.

# When to Use
Before publishing a new release

# Inputs
- Previous and current release tags (e.g., `v0.6.0`, `v0.6.1`)
- Commit log between these tags (`git log v0.6.0...v0.6.1`)
- This template

# Outputs
- Markdown-formatted release notes, matching the sections and style below.

# Workflow
1. Collect all commits between the previous and current release tags.
2. Categorize commits into sections (Features, Bugfixes, Docs, Tests, etc.) based on commit messages.
3. Summarize and group related changes, following the formatting and emoji conventions in the template.
4. List new contributors with links to their GitHub profiles.
5. Generate a changelog table with commit statistics.
6. Output the final release notes in markdown.

# Boundaries
- Do not invent features or changes not present in the commit log.
- Do not include unreleased or future work.
- Do not modify the template structure unless requested.

# Release Notes Template

## Example

```markdown
# Wakis vX.Y.Z

This release introduces major improvements to performance and usability: [short summary of highlights].

## 🚀 New Features

* 🧩 **Component/Module**
  * [Feature description including PR-number]
  * [Feature description]

* 🧩 **Another component/module**
  * [Feature description]

## 💗 Other Tag Highlights

* 🔁 **Tests**
  * [Test-related changes]

* 📚 **Documentation and Examples**
  * [Docs-related changes]
  * [Examples-related changes]
  * [Notebooks-related changes]

* ⚙️ **Build & Compatibility**
  * [Build/compatibility changes]

## 🐛 **Bugfixes**

* [Bugfix description]
* [Bugfix description]

## 👩‍💻 Contributors
* [**@username**](https://github.com/username) — [Contribution summary] (#PR-number)


## 📝 **Full changelog**

| **N commits** | 📚 Docs | 🧪 Tests | 🐛 Fixes | 🎨 Style | ✨ Features | Other |
|---------------|---------|----------|-----------|------------|--------------|-------|
| % of Commits  |  XX%    | XX%      | XX%       | XX%        | XX%          | XX%   |

`git log vX.Y.Z-1...vX.Y.Z --date=short --pretty=format:"* %ad %d %s (%aN)*`

* YYYY-MM-DD  [commit message] ([author])
* ...
```

# Example prompt

"Generate release notes for v0.6.2 using the above template, summarizing all commits since v0.6.1."