#!/bin/bash
set -e
source_root="$(dirname "$(readlink -f "$0")")"
files="$@"
if ! [[ "${files[0]}" ]]; then
	files=($(git diff master --name-only --relative | grep -E 'py$'))
fi
set -x
yapf --parallel --in-place --style "${source_root}"/.style.yapf "${files[@]}"
autopep8 --jobs 0 --in-place --aggressive --aggressive \
	--global-config "${source_root}"/.autopep8-config "${files[@]}"
