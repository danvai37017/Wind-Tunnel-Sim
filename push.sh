#!/bin/bash
set -e
if [ -z "$1" ]; then
  echo "Usage: ./push.sh \"your commit message\""
  exit 1
fi
git add -A
git commit -m "$1" || echo "Nothing to commit."
git push origin main
echo "Push complete."