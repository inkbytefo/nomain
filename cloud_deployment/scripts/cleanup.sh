#!/bin/bash
## Developer: inkbytefo
## Modified: 2025-11-09

set -e

NAMESPACE="chimera-agi"

echo "Cleaning up Chimera AGI deployment..."

# Delete all resources
kubectl delete namespace $NAMESPACE --ignore-not-found=true

# Clean up Docker images
docker rmi chimera-agi:latest 2>/dev/null || true

echo "Cleanup completed"
