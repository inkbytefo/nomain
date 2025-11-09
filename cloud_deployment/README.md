# Chimera AGI Cloud Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying Chimera AGI to cloud environments using Docker and Kubernetes.

## Prerequisites

### System Requirements
- Kubernetes cluster (v1.24+)
- kubectl configured
- Docker installed
- Helm 3.0+ (optional)
- At least 2 CPU cores and 8GB RAM per node
- GPU nodes with NVIDIA drivers (for optimal performance)

### Cloud Provider Setup
The deployment is tested on:
- Google Kubernetes Engine (GKE)
- Amazon EKS
- Azure Kubernetes Service (AKS)

## Quick Start

### 1. Build and Push Docker Image

```bash
# Build the image
./cloud_deployment/scripts/build.sh

# Update the registry in the script and push
# docker push your-registry.com/chimera-agi:latest
```

### 2. Deploy to Kubernetes

```bash
# Deploy to development environment
./cloud_deployment/scripts/deploy.sh development

# Or deploy to production
./cloud_deployment/scripts/deploy.sh production
```

### 3. Test the Deployment

```bash
# Run tests
./cloud_deployment/scripts/test.sh
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHIMERA_ENV` | `development` | Environment name |
| `CHIMERA_CONFIG_PATH` | `/app/config/chimera_cloud.yaml` | Configuration file path |
| `CHIMERA_LOG_LEVEL` | `INFO` | Logging level |

### Resource Requirements

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| Chimera AGI | 2-4 cores | 4-8GB | 100GB |
| Redis | 0.5 cores | 1GB | 10GB |
| Monitoring | 0.5 cores | 1GB | 20GB |

## Monitoring

### Prometheus Metrics

The system exposes the following metrics:
- `neural_spikes_total`: Total neural spikes per layer
- `dopamine_level`: Current dopamine level
- `prediction_error`: Prediction error rate
- `cpu_usage_total`: CPU usage
- `memory_usage_bytes`: Memory usage

### Grafana Dashboard

Access the Grafana dashboard at `http://<service-ip>:3000` with:
- Username: `admin`
- Password: `admin123`

## Scaling

### Horizontal Scaling

```bash
# Scale the deployment
kubectl scale deployment chimera-agi --replicas=4 -n chimera-agi
```

### Resource Scaling

Modify the resource limits in `cloud_deployment/kubernetes/base/deployment.yaml`:

```yaml
resources:
  requests:
    cpu: "4"
    memory: "8Gi"
  limits:
    cpu: "8"
    memory: "16Gi"
```

## Security

### Network Policies

The deployment includes network policies to restrict traffic between components.

### Secrets

Sensitive configuration should be stored in Kubernetes secrets:

```bash
kubectl create secret generic chimera-secrets \
  --from-literal=api-key=your-api-key \
  -n chimera-agi
```

## Troubleshooting

### Common Issues

1. **Pod Pending**: Check resource requests and node availability
2. **CrashLoopBackOff**: Check logs with `kubectl logs`
3. **Service Not Accessible**: Verify service configuration and load balancer

### Debug Commands

```bash
# Check pod status
kubectl get pods -n chimera-agi

# View logs
kubectl logs -f deployment/chimera-agi -n chimera-agi

# Describe pod
kubectl describe pod <pod-name> -n chimera-agi

# Port-forward for local testing
kubectl port-forward service/chimera-agi-service 8000:80 -n chimera-agi
```

## Maintenance

### Updates

```bash
# Update the deployment
kubectl set image deployment/chimera-agi chimera-agi=chimera-agi:v2.0.0 -n chimera-agi

# Rollback if needed
kubectl rollout undo deployment/chimera-agi -n chimera-agi
```

### Backup

```bash
# Backup configuration
kubectl get all -n chimera-agi -o yaml > chimera-backup.yaml

# Backup data
kubectl exec -it deployment/chimera-agi -n chimera-agi -- tar czf /tmp/data-backup.tar.gz /app/data
kubectl cp chimera-agi-<pod-id>:/tmp/data-backup.tar.gz ./data-backup.tar.gz
```

## Cleanup

```bash
# Remove all resources
./cloud_deployment/scripts/cleanup.sh
```

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review pod logs and events
3. Consult the monitoring dashboard
4. Create an issue in the project repository
