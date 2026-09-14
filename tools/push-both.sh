#!/usr/bin/env bash
# Push the current branch to Cursor origin, then to GitHub.
# Origin is required. GitHub is attempted second and reported clearly.
set -euo pipefail

branch="${1:-$(git rev-parse --abbrev-ref HEAD)}"

git push -u origin "$branch"

if git remote get-url github >/dev/null 2>&1; then
  if GIT_TERMINAL_PROMPT=0 git push github "$branch"; then
    echo "Updated GitHub: github/$branch"
  else
    echo "GitHub push failed (no write credentials in this environment)." >&2
    echo "Add a PAT with repo scope, then:" >&2
    echo "  git remote set-url github https://<TOKEN>@github.com/james153dot/goai-semi.git" >&2
    echo "  git push github $branch" >&2
    exit 2
  fi
else
  echo "No 'github' remote configured." >&2
  exit 2
fi
