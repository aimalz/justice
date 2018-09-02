#!/bin/bash
set -e
set -x
yapf --parallel --in-place --style .style.yapf "$@"
autopep8 --jobs 0 --in-place --aggressive --aggressive \
	--global-config .autopep8-config "$@"
