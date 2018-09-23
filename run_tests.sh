#!/bin/bash

# This will make this script become the pre-commit hook for the repository
if [ "$0" != ".git/hooks/pre-commit" ]
    then cp $0 .git/hooks/pre-commit
fi

python -m pytest -s -m "not no_precommit" test "$@"
