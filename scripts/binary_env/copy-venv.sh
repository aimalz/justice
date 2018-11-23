#!/bin/bash
set -x
set -e
(
    cd "$(dirname "$(readlink -f "$0")")"
    docker build -t justice-env .
)
container_id=$(docker create --name justice-env-container justice-env /bin/bash)
docker cp "${container_id}":/home/nonroot/justice-venv-py3.tar.bz2 justice-venv-py3.tar.bz2
docker rm "${container_id}"
