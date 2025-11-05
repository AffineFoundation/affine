#!/bin/bash
# Docker-based Verification Service Manager
# Usage: ./docker-verification.sh [command]

set -e
cd "$(dirname "$0")"

IMAGE="affine-verification:latest"
CONTAINER_NAME="affine-verification"

# Docker run base command
docker_run() {
    docker run --rm \
        -v "$(pwd)/.env:/app/.env:ro" \
        -v "$(pwd)/affine/verification/.env:/app/affine/verification/.env:ro" \
        -v "$HOME/.cache/affine:/root/.cache/affine" \
        -v "/var/run/docker.sock:/var/run/docker.sock" \
        -v "$HOME/.ssh:/root/.ssh:ro" \
        --network host \
        --name "$CONTAINER_NAME-$RANDOM" \
        --entrypoint /opt/venv/bin/af \
        "$IMAGE" "$@"
}

# Docker run in background
docker_run_bg() {
    docker run -d \
        --restart unless-stopped \
        -v "$(pwd)/.env:/app/.env:ro" \
        -v "$(pwd)/affine/verification/.env:/app/affine/verification/.env:ro" \
        -v "$HOME/.cache/affine:/root/.cache/affine" \
        -v "/var/run/docker.sock:/var/run/docker.sock" \
        -v "$HOME/.ssh:/root/.ssh:ro" \
        --network host \
        --name "$CONTAINER_NAME" \
        --entrypoint /opt/venv/bin/af \
        "$IMAGE" "$@"
}

case "${1:-help}" in
    build)
        echo "Building Docker image..."
        docker build -t "$IMAGE" .
        echo "✓ Image built successfully: $IMAGE"
        ;;

    config)
        echo "Showing verification configuration..."
        docker_run verify config
        ;;

    status)
        echo "Checking queue status..."
        docker_run verify status
        ;;

    monitor)
        echo "Running monitor once..."
        docker_run verify monitor --once
        ;;

    worker)
        shift
        DRY_RUN=""
        if [ "$1" = "--dry-run" ]; then
            DRY_RUN="--dry-run"
            echo "Running worker once [DRY-RUN MODE]..."
        else
            echo "Running worker once..."
        fi
        docker_run verify worker --once $DRY_RUN
        ;;

    start)
        shift
        DRY_RUN=""
        if [ "$1" = "--dry-run" ]; then
            DRY_RUN="--dry-run"
            echo "Starting verification service [DRY-RUN MODE]..."
        else
            echo "Starting verification service..."
        fi

        # Stop existing container if running
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true

        # Start new container
        CONTAINER_ID=$(docker_run_bg -vv verify run --monitor-interval 300 --num-workers 1 $DRY_RUN)
        echo "✓ Verification service started"
        echo "  Container ID: $CONTAINER_ID"
        echo "  Container name: $CONTAINER_NAME"
        echo ""
        echo "View logs with: docker logs -f $CONTAINER_NAME"
        ;;

    stop)
        echo "Stopping verification service..."
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
        echo "✓ Service stopped"
        ;;

    restart)
        $0 stop
        sleep 2
        $0 start
        ;;

    logs)
        docker logs -f "$CONTAINER_NAME"
        ;;

    ps)
        echo "Verification containers:"
        docker ps -a --filter "name=$CONTAINER_NAME"
        ;;

    blacklist)
        echo "Showing blacklist..."
        docker_run verify blacklist --show
        ;;

    shell)
        echo "Opening shell in container..."
        docker run --rm -it \
            -v "$(pwd)/.env:/app/.env:ro" \
            -v "$(pwd)/affine/verification/.env:/app/affine/verification/.env:ro" \
            -v "$HOME/.cache/affine:/root/.cache/affine" \
            -v "/var/run/docker.sock:/var/run/docker.sock" \
            -v "$HOME/.ssh:/root/.ssh:ro" \
            --network host \
            --entrypoint /bin/bash \
            "$IMAGE"
        ;;

    help|*)
        cat <<EOF
Docker-based Verification Service Manager

Usage: $0 [command] [options]

Build Commands:
  build           Build Docker image

Management Commands:
  start           Start verification service in background
  start --dry-run Start in dry-run mode (no R2 uploads)
  stop            Stop verification service
  restart         Restart verification service
  logs            View service logs (follow mode)
  ps              Show container status

Testing Commands:
  config          Show configuration
  status          Show queue status
  monitor         Run monitor once
  worker          Run worker once
  worker --dry-run Run worker in dry-run mode
  blacklist       Show blacklist
  shell           Open interactive shell in container

Examples:
  $0 build                 # Build image
  $0 config                # Check configuration
  $0 status                # Check queue
  $0 worker --dry-run      # Test worker
  $0 start                 # Start service
  $0 logs                  # View logs
  $0 stop                  # Stop service

EOF
        ;;
esac
