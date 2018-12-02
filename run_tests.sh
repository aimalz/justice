#!/bin/bash

# This will make this script become the pre-commit hook for the repository
if [ "$0" != ".git/hooks/pre-commit" ]
    then cp $0 .git/hooks/pre-commit
fi

set -e
mypy --ignore-missing-imports justice/*.py
python -m pytest -s -m "not slow" test "$@"
if [ "$0" != ".git/hooks/pre-commit" ]; then
    python -m pytest -s -m "slow" test "$@"
else
    echo "Not running slow test for pre-commit, please run ./run-tests.sh manually." >&2
fi
