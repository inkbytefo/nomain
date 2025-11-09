#!/bin/bash
## Developer: inkbytefo
## Modified: 2025-11-09

set -e

NAMESPACE="chimera-agi"
SERVICE_NAME="chimera-agi-service"

echo "Testing Chimera AGI deployment..."

# Get service URL
SERVICE_URL=$(kubectl get service $SERVICE_NAME -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

if [ -z "$SERVICE_URL" ]; then
    echo "Service URL not found. Using port-forward..."
    kubectl port-forward service/$SERVICE_NAME -n $NAMESPACE 8000:80 &
    PORT_FORWARD_PID=$!
    sleep 5
    SERVICE_URL="localhost:8000"
fi

# Health check
echo "Checking health endpoint..."
curl -f http://$SERVICE_URL/health || {
    echo "Health check failed"
    exit 1
}

# Test API endpoints
echo "Testing API endpoints..."
curl -f http://$SERVICE_URL/status || {
    echo "Status check failed"
    exit 1
}

# Load test
echo "Running load test..."
for i in {1..10}; do
    curl -s http://$SERVICE_URL/status > /dev/null
done

echo "All tests passed successfully"

# Clean up port-forward
if [ ! -z "$PORT_FORWARD_PID" ]; then
    kill $PORT_FORWARD_PID
fi
