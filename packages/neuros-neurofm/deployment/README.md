# NeuroFM-X Deployment Guide

Complete guide for deploying NeuroFM-X in production environments.

## Table of Contents

1. [Docker Deployment](#docker-deployment)
2. [AWS Deployment](#aws-deployment)
3. [GCP Deployment](#gcp-deployment)
4. [Azure Deployment](#azure-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Monitoring & Logging](#monitoring--logging)

---

## Docker Deployment

### Basic Deployment

```bash
# Build image
docker build -t neurofm-x:latest .

# Run container (CPU)
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  neurofm-x:latest

# Run container (GPU)
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -e NEUROFM_DEVICE=cuda \
  neurofm-x:latest
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# Start with GPU support
docker-compose --profile gpu up -d

# Start with monitoring
docker-compose --profile monitoring up -d

# View logs
docker-compose logs -f neurofm-api

# Stop services
docker-compose down
```

---

## AWS Deployment

### Option 1: EC2 Instance

**Step 1: Launch EC2 Instance**

```bash
# Create EC2 instance (GPU instance: p3.2xlarge, p4d.24xlarge)
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type p3.2xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-0123456789abcdef0 \
  --subnet-id subnet-0123456789abcdef0 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=neurofm-x}]'
```

**Step 2: Install Docker & NVIDIA Container Toolkit**

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<instance-ip>

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install NVIDIA Container Toolkit (for GPU)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Step 3: Deploy NeuroFM-X**

```bash
# Clone repository
git clone https://github.com/your-org/neuros-v1.git
cd neuros-v1/packages/neuros-neurofm

# Build and run
docker-compose --profile gpu up -d
```

### Option 2: ECS (Elastic Container Service)

**Step 1: Push Image to ECR**

```bash
# Create ECR repository
aws ecr create-repository --repository-name neurofm-x

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push image
docker tag neurofm-x:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/neurofm-x:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/neurofm-x:latest
```

**Step 2: Create ECS Task Definition**

Use the provided `aws-ecs-task-definition.json` file:

```bash
aws ecs register-task-definition \
  --cli-input-json file://deployment/aws-ecs-task-definition.json
```

**Step 3: Create ECS Service**

```bash
aws ecs create-service \
  --cluster neurofm-cluster \
  --service-name neurofm-service \
  --task-definition neurofm-x:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
```

### Option 3: SageMaker

Use `deploy_sagemaker.py` script:

```bash
python deployment/deploy_sagemaker.py \
  --model-path models/neurofmx.pt \
  --instance-type ml.p3.2xlarge \
  --endpoint-name neurofm-x-endpoint
```

---

## GCP Deployment

### Option 1: Compute Engine

**Step 1: Create GPU Instance**

```bash
gcloud compute instances create neurofm-x-instance \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE \
  --metadata=startup-script='#!/bin/bash
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
      sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker'
```

**Step 2: Deploy NeuroFM-X**

```bash
# SSH into instance
gcloud compute ssh neurofm-x-instance

# Clone and deploy
git clone https://github.com/your-org/neuros-v1.git
cd neuros-v1/packages/neuros-neurofm
docker-compose --profile gpu up -d
```

### Option 2: Cloud Run

**Step 1: Push to Container Registry**

```bash
# Tag and push
docker tag neurofm-x:latest gcr.io/<project-id>/neurofm-x:latest
docker push gcr.io/<project-id>/neurofm-x:latest
```

**Step 2: Deploy to Cloud Run**

```bash
gcloud run deploy neurofm-x \
  --image gcr.io/<project-id>/neurofm-x:latest \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4 \
  --port 8000 \
  --allow-unauthenticated
```

---

## Azure Deployment

### Option 1: Azure Container Instances

```bash
# Create resource group
az group create --name neurofm-rg --location eastus

# Create container
az container create \
  --resource-group neurofm-rg \
  --name neurofm-x \
  --image neurofmx/neurofm-x:latest \
  --cpu 4 \
  --memory 8 \
  --ports 8000 \
  --environment-variables NEUROFM_DEVICE=cpu \
  --dns-name-label neurofm-x
```

### Option 2: Azure Kubernetes Service (AKS)

See [Kubernetes Deployment](#kubernetes-deployment) section.

---

## Kubernetes Deployment

### Step 1: Create Kubernetes Manifests

Use the provided manifests in `deployment/k8s/`:

- `deployment.yaml` - Main deployment
- `service.yaml` - Load balancer service
- `configmap.yaml` - Configuration
- `hpa.yaml` - Horizontal Pod Autoscaler

### Step 2: Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace neurofm

# Apply manifests
kubectl apply -f deployment/k8s/ -n neurofm

# Check status
kubectl get pods -n neurofm
kubectl get svc -n neurofm

# View logs
kubectl logs -f deployment/neurofm-x -n neurofm
```

### Step 3: Autoscaling

```bash
# Apply HPA
kubectl apply -f deployment/k8s/hpa.yaml -n neurofm

# Check HPA status
kubectl get hpa -n neurofm
```

---

## Monitoring & Logging

### Prometheus + Grafana

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Access Grafana
open http://localhost:3000
# Login: admin / admin

# Access Prometheus
open http://localhost:9090
```

### CloudWatch (AWS)

```bash
# Enable CloudWatch Container Insights
aws ecs update-cluster-settings \
  --cluster neurofm-cluster \
  --settings name=containerInsights,value=enabled
```

### Stackdriver (GCP)

```bash
# Logs are automatically sent to Cloud Logging
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=neurofm-x"
```

### Application Insights (Azure)

```bash
# Add to container environment variables
--environment-variables APPLICATIONINSIGHTS_CONNECTION_STRING=<connection-string>
```

---

## Performance Tuning

### CPU Optimization

```bash
# Use optimized batch size
docker run -e NEUROFM_BATCH_SIZE=64 -e NEUROFM_MAX_WAIT_MS=20.0 neurofm-x:latest
```

### GPU Optimization

```bash
# Enable mixed precision
docker run --gpus all -e NEUROFM_DEVICE=cuda -e NEUROFM_PRECISION=fp16 neurofm-x:latest
```

### Model Compression

```bash
# Use quantized model
python -m neuros_neurofm.optimization.quantize_model \
  --model-path models/neurofmx.pt \
  --output-path models/neurofmx_int8.pt \
  --quantization-type dynamic

# Deploy quantized model
docker run -e NEUROFM_MODEL_PATH=/app/models/neurofmx_int8.pt neurofm-x:latest
```

---

## Security Best Practices

1. **Use secrets management**: AWS Secrets Manager, GCP Secret Manager, Azure Key Vault
2. **Enable authentication**: Add API keys or OAuth2
3. **Use HTTPS**: Deploy behind a reverse proxy (nginx, Traefik)
4. **Network isolation**: Use VPC/VNet, security groups
5. **Regular updates**: Keep base images and dependencies updated

---

## Troubleshooting

### High Latency

- Increase batch size: `NEUROFM_BATCH_SIZE=128`
- Reduce wait time: `NEUROFM_MAX_WAIT_MS=5.0`
- Use GPU: `NEUROFM_DEVICE=cuda`
- Enable model compression

### Out of Memory

- Reduce batch size: `NEUROFM_BATCH_SIZE=16`
- Use smaller model variant
- Enable gradient checkpointing
- Use mixed precision

### Model Loading Errors

- Check model path: `NEUROFM_MODEL_PATH=/app/models/model.pt`
- Verify model format (TorchScript)
- Check PyTorch version compatibility

---

## Support

For issues and questions:
- GitHub Issues: https://github.com/your-org/neuros-v1/issues
- Documentation: https://neuros.readthedocs.io
- Email: support@neuros.ai
