#!/bin/bash

# Push script for NeuroFMx training container

set -e

# Configuration
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"ghcr.io"}
DOCKER_USERNAME=${DOCKER_USERNAME:-"YOUR_USERNAME"}  # Change this!
IMAGE_NAME="neurofmx-train"
IMAGE_TAG=${IMAGE_TAG:-"latest"}
FULL_IMAGE="${DOCKER_REGISTRY}/${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "═══════════════════════════════════════════════════════"
echo "Pushing NeuroFMx Training Container"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Image: ${FULL_IMAGE}"
echo ""

# Login to registry if needed
if [[ "${DOCKER_REGISTRY}" == "ghcr.io" ]]; then
    echo "Logging in to GitHub Container Registry..."
    echo "Make sure you have set GITHUB_TOKEN environment variable"
    echo "${GITHUB_TOKEN}" | docker login ghcr.io -u "${DOCKER_USERNAME}" --password-stdin
fi

echo "Pushing image..."
docker push "${FULL_IMAGE}"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "Push complete!"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Image available at: ${FULL_IMAGE}"
echo ""
echo "Next steps:"
echo "  1. Update K8s manifest with image: ${FULL_IMAGE}"
echo "  2. Deploy to cluster: cd infra && make k8s-apply"
