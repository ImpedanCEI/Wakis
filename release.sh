#!/bin/bash
set -euo pipefail; IFS=$'\n\t'

NAME=$( python setup.py --name )
VER=$( python setup.py --version )

echo "========================================================================"
echo "Tagging $NAME v$VER"
echo "========================================================================"

git tag -a v$VER
git push origin v$VER