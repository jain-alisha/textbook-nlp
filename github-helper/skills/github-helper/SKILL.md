---
name: github-helper
description: Analyze and manage GitHub repositories, issues, pull requests, and CI workflows.
---

# GitHub Helper

Use the GitHub MCP server whenever a request requires information from GitHub.

## Repository analysis

When analyzing a repository:

1. Inspect the repository structure.
2. Identify the files most relevant to the user's question.
3. Read those files before drawing conclusions.
4. Trace imports and dependencies when necessary.
5. Cite filenames and relevant code locations in the answer.

## Pull requests

When reviewing a pull request:

1. Understand the purpose of the PR.
2. Inspect the changed files.
3. Check for correctness problems.
4. Look for regressions and edge cases.
5. Separate blocking issues from optional improvements.

Do not modify or merge a pull request unless the user explicitly requests that action.

## Issues

When investigating an issue:

1. Read the issue and relevant discussion.
2. Locate the likely code involved.
3. Determine the most likely cause.
4. Suggest a concrete fix.
5. Identify which files would need modification.

## CI failures

When asked about failing CI:

1. Inspect the workflow run.
2. Find the first meaningful error rather than reporting cascading failures.
3. Trace that error to the relevant code or configuration.
4. Explain the cause.
5. Suggest the smallest reasonable fix.

## Safety and scope

Prefer read-only actions unless the user explicitly asks to modify or create GitHub resources. Keep actions narrow, explain the reasoning, and avoid destructive changes.
