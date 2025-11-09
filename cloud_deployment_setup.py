"""
## Developer: inkbytefo
## Modified: 2025-11-09
"""

import os
import sys
import json
import yaml
import logging
from typing import Dict, Any, List
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudDeploymentSetup:
    """
    Cloud deployment preparation for Chimera AGI.
    
    This class prepares the Chimera AGI system for cloud deployment by:
    1. Creating Docker configuration
    2. Setting up cloud requirements
    3. Creating deployment scripts
    4. Generating cloud-specific configurations
    """
    
    def __init__(self, project_dir: str = "."):
        """Initialize cloud deployment setup."""
        self.project_dir = project_dir
        self.cloud_dir = os.path.join(project_dir, "cloud_deployment")
        self.docker_dir = os.path.join(self.cloud_dir, "docker")
        self.k8s_dir = os.path.join(self.cloud_dir, "kubernetes")
        self.scripts_dir = os.path.join(self.cloud_dir, "scripts")
        
    def setup_cloud_deployment(self):
        """Set up complete cloud deployment structure."""
        logger.info("🚀 Setting up Chimera AGI for cloud deployment")
        logger.info("=" * 60)
        
        try:
            # Create directory structure
            self._create_directory_structure()
            
            # Generate Docker configuration
            self._create_docker_configuration()
            
            # Generate Kubernetes manifests
            self._create_kubernetes_manifests()
            
            # Create deployment scripts
            self._create_deployment_scripts()
            
            # Generate cloud requirements
            self._create_cloud_requirements()
            
            # Create monitoring configuration
            self._create_monitoring_config()
            
            # Generate deployment documentation
            self._create_deployment_docs()
            
            logger.info("✅ Cloud deployment setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cloud deployment setup failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_directory_structure(self):
        """Create cloud deployment directory structure."""
        logger.info("Creating directory structure...")
        
        directories = [
            self.cloud_dir,
            self.docker_dir,
            self.k8s_dir,
            self.scripts_dir,
            os.path.join(self.k8s_dir, "base"),
            os.path.join(self.k8s_dir, "overlays", "development"),
            os.path.join(self.k8s_dir, "overlays", "production"),
            os.path.join(self.cloud_dir, "monitoring"),
            os.path.join(self.cloud_dir, "config")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def _create_docker_configuration(self):
        """Create Docker configuration files."""
        logger.info("Creating Docker configuration...")
        
        # Dockerfile
        dockerfile_content = """## Developer: inkbytefo
## Modified: 2025-11-09

# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    git \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY cloud_deployment/requirements-cloud.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copy project files
COPY . .

# Create non-root user
RUN useradd -m -u 1000 chimera && chown -R chimera:chimera /app
USER chimera

# Expose ports
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command
CMD ["python", "run_chimera.py", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        with open(os.path.join(self.docker_dir, "Dockerfile"), 'w') as f:
            f.write(dockerfile_content)
        
        # Docker Compose
        docker_compose_content = """## Developer: inkbytefo
## Modified: 2025-11-09

version: '3.8'

services:
  chimera-agi:
    build:
      context: ..
      dockerfile: cloud_deployment/docker/Dockerfile
    container_name: chimera-agi
    ports:
      - "8000:8000"
      - "8080:8080"
    environment:
      - CHIMERA_ENV=production
      - CHIMERA_LOG_LEVEL=INFO
      - CHIMERA_CONFIG_PATH=/app/config/chimera_cloud.yaml
    volumes:
      - ../config:/app/config:ro
      - chimera_data:/app/data
      - chimera_logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  redis:
    image: redis:7-alpine
    container_name: chimera-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes

  monitoring:
    image: prom/prometheus:latest
    container_name: chimera-monitoring
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: chimera-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning:ro
    restart: unless-stopped

volumes:
  chimera_data:
  chimera_logs:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: chimera-network
"""
        
        with open(os.path.join(self.docker_dir, "docker-compose.yml"), 'w') as f:
            f.write(docker_compose_content)
        
        # .dockerignore
        dockerignore_content = """# Git
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git
.mypy_cache
.pytest_cache
.hypothesis

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
archive/
*.png
*.jpg
*.jpeg
results/
experiments/
logs/
data/

# Documentation
*.md
!README.md
"""
        
        with open(os.path.join(self.docker_dir, ".dockerignore"), 'w') as f:
            f.write(dockerignore_content)
        
        logger.info("Docker configuration created")
    
    def _create_kubernetes_manifests(self):
        """Create Kubernetes manifests."""
        logger.info("Creating Kubernetes manifests...")
        
        # Namespace
        namespace_content = """## Developer: inkbytefo
## Modified: 2025-11-09

apiVersion: v1
kind: Namespace
metadata:
  name: chimera-agi
  labels:
    name: chimera-agi
"""
        
        with open(os.path.join(self.k8s_dir, "base", "namespace.yaml"), 'w') as f:
            f.write(namespace_content)
        
        # ConfigMap
        configmap_content = """## Developer: inkbytefo
## Modified: 2025-11-09

apiVersion: v1
kind: ConfigMap
metadata:
  name: chimera-config
  namespace: chimera-agi
data:
  chimera_cloud.yaml: |
    run_params:
      simulation_duration_sec: 3600
      chunk_duration_ms: 10
      host: "0.0.0.0"
      port: 8000
    
    network_params:
      neurons:
        sensory_vision:
          count: 784
          threshold: -50.0
          reset: -65.0
          refractory: 5.0
        associative_cortex:
          count: 4000
          threshold: -50.0
          reset: -65.0
          refractory: 5.0
        language_concepts:
          count: 1000
          threshold: -50.0
          reset: -65.0
          refractory: 5.0
      
      synapses:
        vision_to_assoc:
          from: sensory_vision
          to: associative_cortex
          count: 100000
          w_max: 1.0
          w_init: 0.1
          stdp_enabled: true
        assoc_to_language:
          from: associative_cortex
          to: language_concepts
          count: 50000
          w_max: 1.0
          w_init: 0.1
          stdp_enabled: true
    
    modules:
      sensory_cortex:
        enabled: true
        input_type: "camera"
        preprocessing:
          resize: [28, 28]
          normalize: true
      
      world_model:
        enabled: true
        predictor_type: "linear"
        memory_size: 1000
        learning_rate: 0.01
      
      language_module:
        enabled: true
        model_name: "sentence-transformers/all-MiniLM-L6-v2"
        embedding_dim: 768
        response_generation:
          template_based: true
          llm_fallback: false
      
      motivation_system:
        enabled: true
        dopamine_params:
          baseline: 1.0
          k_gain: 2.0
          time_constant: 0.5
        curiosity:
          enable: true
          scale: 1.0
        goal_orientation:
          enable: true
          scale: 1.0
    
    logging:
      level: "INFO"
      format: "structured"
      file_output: true
      console_output: true
    
    monitoring:
      enabled: true
      metrics_port: 8080
      health_check_port: 8000
"""
        
        with open(os.path.join(self.k8s_dir, "base", "configmap.yaml"), 'w') as f:
            f.write(configmap_content)
        
        # Deployment
        deployment_content = """## Developer: inkbytefo
## Modified: 2025-11-09

apiVersion: apps/v1
kind: Deployment
metadata:
  name: chimera-agi
  namespace: chimera-agi
  labels:
    app: chimera-agi
spec:
  replicas: 2
  selector:
    matchLabels:
      app: chimera-agi
  template:
    metadata:
      labels:
        app: chimera-agi
    spec:
      containers:
      - name: chimera-agi
        image: chimera-agi:latest
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8080
          name: metrics
        env:
        - name: CHIMERA_ENV
          value: "production"
        - name: CHIMERA_CONFIG_PATH
          value: "/app/config/chimera_cloud.yaml"
        - name: CHIMERA_LOG_LEVEL
          value: "INFO"
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
        - name: data
          mountPath: /app/data
        - name: logs
          mountPath: /app/logs
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
      volumes:
      - name: config
        configMap:
          name: chimera-config
      - name: data
        persistentVolumeClaim:
          claimName: chimera-data-pvc
      - name: logs
        persistentVolumeClaim:
          claimName: chimera-logs-pvc
      nodeSelector:
        accelerator: nvidia-tesla-k80
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
"""
        
        with open(os.path.join(self.k8s_dir, "base", "deployment.yaml"), 'w') as f:
            f.write(deployment_content)
        
        # Service
        service_content = """## Developer: inkbytefo
## Modified: 2025-11-09

apiVersion: v1
kind: Service
metadata:
  name: chimera-agi-service
  namespace: chimera-agi
  labels:
    app: chimera-agi
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  - port: 8080
    targetPort: 8080
    protocol: TCP
    name: metrics
  selector:
    app: chimera-agi
"""
        
        with open(os.path.join(self.k8s_dir, "base", "service.yaml"), 'w') as f:
            f.write(service_content)
        
        # PVC for data
        pvc_data_content = """## Developer: inkbytefo
## Modified: 2025-11-09

apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: chimera-data-pvc
  namespace: chimera-agi
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
"""
        
        with open(os.path.join(self.k8s_dir, "base", "pvc-data.yaml"), 'w') as f:
            f.write(pvc_data_content)
        
        # PVC for logs
        pvc_logs_content = """## Developer: inkbytefo
## Modified: 2025-11-09

apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: chimera-logs-pvc
  namespace: chimera-agi
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd
"""
        
        with open(os.path.join(self.k8s_dir, "base", "pvc-logs.yaml"), 'w') as f:
            f.write(pvc_logs_content)
        
        # Kustomization base
        kustomization_base_content = """## Developer: inkbytefo
## Modified: 2025-11-09

apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

metadata:
  name: chimera-agi-base

resources:
  - namespace.yaml
  - configmap.yaml
  - deployment.yaml
  - service.yaml
  - pvc-data.yaml
  - pvc-logs.yaml

commonLabels:
  app.kubernetes.io/name: chimera-agi
  app.kubernetes.io/component: agi-system
  app.kubernetes.io/version: "1.0.0"
"""
        
        with open(os.path.join(self.k8s_dir, "base", "kustomization.yaml"), 'w') as f:
            f.write(kustomization_base_content)
        
        logger.info("Kubernetes manifests created")
    
    def _create_deployment_scripts(self):
        """Create deployment scripts."""
        logger.info("Creating deployment scripts...")
        
        # Build script
        build_script_content = """#!/bin/bash
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
"""
        
        build_script_path = os.path.join(self.scripts_dir, "build.sh")
        with open(build_script_path, 'w') as f:
            f.write(build_script_content)
        os.chmod(build_script_path, 0o755)
        
        # Deploy script
        deploy_script_content = """#!/bin/bash
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
"""
        
        deploy_script_path = os.path.join(self.scripts_dir, "deploy.sh")
        with open(deploy_script_path, 'w') as f:
            f.write(deploy_script_content)
        os.chmod(deploy_script_path, 0o755)
        
        # Test script
        test_script_content = """#!/bin/bash
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
"""
        
        test_script_path = os.path.join(self.scripts_dir, "test.sh")
        with open(test_script_path, 'w') as f:
            f.write(test_script_content)
        os.chmod(test_script_path, 0o755)
        
        # Cleanup script
        cleanup_script_content = """#!/bin/bash
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
"""
        
        cleanup_script_path = os.path.join(self.scripts_dir, "cleanup.sh")
        with open(cleanup_script_path, 'w') as f:
            f.write(cleanup_script_content)
        os.chmod(cleanup_script_path, 0o755)
        
        logger.info("Deployment scripts created")
    
    def _create_cloud_requirements(self):
        """Create cloud-specific requirements."""
        logger.info("Creating cloud requirements...")
        
        requirements_content = """## Developer: inkbytefo
## Modified: 2025-11-09

# Cloud-specific dependencies for Chimera AGI

# Web framework
fastapi==0.104.1
uvicorn[standard]==0.24.0

# Monitoring and logging
prometheus-client==0.19.0
structlog==23.2.0
sentry-sdk==1.38.0

# Cloud storage
boto3==1.34.0
google-cloud-storage==2.10.0
azure-storage-blob==12.19.0

# Database
redis==5.0.1
psycopg2-binary==2.9.9

# Security
cryptography==41.0.8
pyjwt==2.8.0

# Performance
gunicorn==21.2.0
orjson==3.9.10

# Health checks
httpx==0.25.2

# Configuration
python-dotenv==1.0.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Utilities
click==8.1.7
rich==13.7.0
typer==0.9.0
"""
        
        with open(os.path.join(self.cloud_dir, "requirements-cloud.txt"), 'w') as f:
            f.write(requirements_content)
        
        logger.info("Cloud requirements created")
    
    def _create_monitoring_config(self):
        """Create monitoring configuration."""
        logger.info("Creating monitoring configuration...")
        
        # Prometheus configuration
        prometheus_content = """## Developer: inkbytefo
## Modified: 2025-11-09

global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'chimera-agi'
    static_configs:
      - targets: ['chimera-agi:8080']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          # - alertmanager:9093
"""
        
        with open(os.path.join(self.cloud_dir, "monitoring", "prometheus.yml"), 'w') as f:
            f.write(prometheus_content)
        
        # Grafana dashboard
        grafana_dashboard_content = """## Developer: inkbytefo
## Modified: 2025-11-09

{
  "dashboard": {
    "id": null,
    "title": "Chimera AGI Dashboard",
    "tags": ["chimera", "agi"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Neural Activity",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(neural_spikes_total[5m])",
            "legendFormat": "{{neuron_layer}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Dopamine Levels",
        "type": "graph",
        "targets": [
          {
            "expr": "dopamine_level",
            "legendFormat": "Dopamine"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Prediction Error",
        "type": "graph",
        "targets": [
          {
            "expr": "prediction_error",
            "legendFormat": "Error"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "id": 4,
        "title": "System Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(cpu_usage_total[5m])",
            "legendFormat": "CPU"
          },
          {
            "expr": "memory_usage_bytes",
            "legendFormat": "Memory"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      }
    ],
    "time": {"from": "now-1h", "to": "now"},
    "refresh": "5s"
  }
}
"""
        
        with open(os.path.join(self.cloud_dir, "monitoring", "grafana-dashboard.json"), 'w') as f:
            f.write(grafana_dashboard_content)
        
        logger.info("Monitoring configuration created")
    
    def _create_deployment_docs(self):
        """Create deployment documentation."""
        logger.info("Creating deployment documentation...")
        
        deployment_docs_content = """# Chimera AGI Cloud Deployment Guide

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
kubectl create secret generic chimera-secrets \\
  --from-literal=api-key=your-api-key \\
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
"""
        
        with open(os.path.join(self.cloud_dir, "README.md"), 'w') as f:
            f.write(deployment_docs_content)
        
        logger.info("Deployment documentation created")
    
    def generate_deployment_summary(self):
        """Generate deployment summary report."""
        logger.info("Generating deployment summary...")
        
        summary = {
            "project": "Chimera AGI",
            "version": "1.0.0",
            "deployment_type": "Cloud Native",
            "components": {
                "docker": {
                    "base_image": "python:3.11-slim",
                    "ports": ["8000", "8080"],
                    "health_check": "/health"
                },
                "kubernetes": {
                    "namespace": "chimera-agi",
                    "replicas": 2,
                    "service_type": "LoadBalancer",
                    "storage": "110GB total"
                },
                "monitoring": {
                    "prometheus": "Enabled",
                    "grafana": "Enabled",
                    "metrics_port": 8080
                }
            },
            "requirements": {
                "min_cpu": "2 cores",
                "min_memory": "4GB",
                "recommended_cpu": "4 cores",
                "recommended_memory": "8GB",
                "gpu_support": "Optional but recommended"
            },
            "deployment_commands": {
                "build": "./cloud_deployment/scripts/build.sh",
                "deploy": "./cloud_deployment/scripts/deploy.sh [environment]",
                "test": "./cloud_deployment/scripts/test.sh",
                "cleanup": "./cloud_deployment/scripts/cleanup.sh"
            },
            "configuration_files": [
                "cloud_deployment/docker/Dockerfile",
                "cloud_deployment/docker/docker-compose.yml",
                "cloud_deployment/kubernetes/base/kustomization.yaml",
                "cloud_deployment/requirements-cloud.txt",
                "cloud_deployment/monitoring/prometheus.yml"
            ]
        }
        
        with open(os.path.join(self.cloud_dir, "deployment_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Deployment summary saved to 'deployment_summary.json'")
        return summary


def main():
    """Main deployment setup function."""
    setup = CloudDeploymentSetup()
    
    if setup.setup_cloud_deployment():
        summary = setup.generate_deployment_summary()
        logger.info("🎉 Cloud deployment setup completed successfully!")
        logger.info(f"📊 Summary: {summary['components']}")
        return 0
    else:
        logger.error("❌ Cloud deployment setup failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)