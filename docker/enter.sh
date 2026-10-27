#!/bin/bash -e
# Open a shell in a persistent copy of the setup.sh container. setup.sh's --rm container reruns
# entrypoint.sh (pip install, apt gcc-12) on every start; this one runs it only when the container is
# created (and once more after a reboot), then each call just opens a new shell. Edits on the host show up
# inside immediately through the /uvm-eviction mount. `docker rm -f uvm-eviction` starts over fresh.
cd "$(dirname "$0")"
IMAGE=cuda-container
NAME=uvm-eviction

# the container mounts this kernel's headers, so a host kernel update needs a new container
if docker container inspect $NAME >/dev/null 2>&1 &&
   [ "$(docker inspect -f '{{index .Config.Labels "kernel"}}' $NAME)" != "$(uname -r)" ]; then
  echo "Host kernel changed; recreating $NAME"
  docker rm -f $NAME >/dev/null
fi

if ! docker container inspect $NAME >/dev/null 2>&1; then
  if ! docker image inspect $IMAGE >/dev/null 2>&1; then
    ./check-kernel-headers.sh
    ./ngc_login.sh
    docker build . -t $IMAGE
  fi
  docker run -d --name $NAME --label kernel=$(uname -r) \
    --init --hostname=cuda-host --privileged=true --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$(pwd)/../:/uvm-eviction" -w /uvm-eviction \
    -v /lib/modules:/lib/modules \
    -v /usr/src/:/usr/src:ro \
    -v /usr/src/kernels/$(uname -r):/usr/src/kernels/$(uname -r):ro \
    $IMAGE sleep infinity >/dev/null
elif [ "$(docker inspect -f '{{.State.Running}}' $NAME)" != "true" ]; then
  docker start $NAME >/dev/null
fi

# wait for entrypoint.sh to finish (instant when it already has)
echo "Waiting for $NAME to finish setup (first start takes a few minutes)..."
until docker logs --since "$(docker inspect -f '{{.State.StartedAt}}' $NAME)" $NAME 2>&1 | grep -q "Environment complete"; do
  if [ "$(docker inspect -f '{{.State.Running}}' $NAME)" != "true" ]; then
    echo "$NAME exited during setup; see: docker logs $NAME"
    exit 1
  fi
  sleep 2
done

# umask 000 as in entrypoint.sh, so files made as root stay writable from the host
exec docker exec -it $NAME bash -c 'umask 000; exec bash'
