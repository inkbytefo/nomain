#!/bin/bash
## Developer: inkbytefo
## Modified: 2025-11-09

set -e

echo "Building Chimera AGI for cloud deployment..."

# Build Docker image
docker build -f cloud_deployment/docker/Dockerfile -t chimera-agi:latest .

# Tag for registry (replace with your registry)
REGISTRY="your-registry.com"
docker tag chimera-agi:latest $REGISTRY/chimera-agi:latest

# Push to registry (uncomment when ready)
# docker push $REGISTRY/chimera-agi:latest

echo "Build completed successfully"
