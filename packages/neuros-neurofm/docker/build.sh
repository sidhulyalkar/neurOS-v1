#!/bin/bash

# Build script for NeuroFMx training container

set -e

# Configuration
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"ghcr.io"}
DOCKER_USERNAME=${DOCKER_USERNAME:-"YOUR_USERNAME"}  # Change this!
IMAGE_NAME="neurofmx-train"
IMAGE_TAG=${IMAGE_TAG:-"latest"}
FULL_IMAGE="${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "═══════════════════════════════════════════════════════"
echo "Building NeuroFMx Training Container"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Image: ${FULL_IMAGE}"
echo ""

# Build from the neuros-neurofm directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.."

echo "Building Docker image..."
docker build \
    -f docker/Dockerfile \
    -t "${FULL_IMAGE}" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

echo ""
echo "═══════════════════════════════════════════════════════"
echo "Build complete!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Image: ${FULL_IMAGE}"
echo ""
echo "Next steps:"
echo "  1. Test locally: docker run --gpus all -it ${FULL_IMAGE}"
echo "  2. Push to registry: docker push ${FULL_IMAGE}"
echo "  3. Update K8s manifest: infra/k8s/20-raycluster-neurofmx.yaml"
