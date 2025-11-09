#!/bin/bash
## Developer: inkbytefo
## Modified: 2025-11-09

set -e

ENVIRONMENT=${1:-development}
NAMESPACE="chimera-agi"

echo "Deploying Chimera AGI to $ENVIRONMENT environment..."

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply configuration
kubectl apply -k cloud_deployment/kubernetes/overlays/$ENVIRONMENT

# Wait for deployment
echo "Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/chimera-agi -n $NAMESPACE

# Show status
echo "Deployment status:"
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

echo "Deployment completed successfully"
echo "Service URL: $(kubectl get service chimera-agi-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"
