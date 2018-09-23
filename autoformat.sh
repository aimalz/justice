#!/bin/bash
set -e
files="$@"
if ! [[ "${files[0]}" ]]; then
	files=($(git diff master --name-only --relative | grep -E 'py$'))
fi
set -x
yapf --parallel --in-place --style .style.yapf "${files[@]}"
autopep8 --jobs 0 --in-place --aggressive --aggressive \
	--global-config .autopep8-config "${files[@]}"
