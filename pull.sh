#!/bin/bash
set -e
echo "Fetching latest from GitHub..."
git fetch origin
echo "Overwriting local files with GitHub's main branch..."
git reset --hard origin/main
chmod +x deploy.sh pull.sh push.sh
echo "Pull complete. Local files now match GitHub exactly."