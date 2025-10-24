# NeuroFMx Cloud Deployment Guide

Complete infrastructure-as-code for deploying NeuroFMx on H100 HGX clusters using Terraform + Kubernetes + Ray.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Cloud Provider Setup](#cloud-provider-setup)
  - [CoreWeave (Recommended)](#coreweave-recommended)
  - [Crusoe Cloud (Alternative)](#crusoe-cloud-alternative)
- [Docker Image Setup](#docker-image-setup)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Training Jobs](#training-jobs)
- [Monitoring & Debugging](#monitoring--debugging)
- [Cost Optimization](#cost-optimization)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This infrastructure deploys NeuroFMx for multimodal neural foundation model training on:

- **Hardware**: 8x H100 80GB HGX with NVLink/NVSwitch
- **Orchestration**: Kubernetes + Ray for distributed training
- **Storage**: S3-compatible object storage for checkpoints
- **Monitoring**: Ray Dashboard + WandB integration

**Supported Providers**:
- ✅ **CoreWeave** (Primary) - Managed Kubernetes (CKS) with official Terraform provider
- ✅ **Crusoe Cloud** (Alternative) - Self-managed K3s on H100 instances

## 🔧 Prerequisites

### Local Tools

```bash
# Install Terraform
brew install terraform  # macOS
# OR
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip

# Install kubectl
brew install kubectl  # macOS
# OR
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# Install Docker
# Follow: https://docs.docker.com/get-docker/

# Verify installations
terraform version  # Should be >= 1.6
kubectl version --client
docker --version
```

### Cloud Provider Accounts

**CoreWeave**:
- Sign up: https://cloud.coreweave.com/
- Get API credentials: https://docs.coreweave.com/coreweave-kubernetes/getting-started

**Crusoe**:
- Sign up: https://console.crusoecloud.com/
- Get API token: https://docs.crusoecloud.com/quickstart/authentication

### Container Registry

Choose one:
- **GitHub Container Registry (ghcr.io)** - Free for public repos
- **Docker Hub**
- **AWS ECR**
- **Google GCR**

## 🚀 Quick Start

### Option 1: CoreWeave (Fastest)

```bash
# 1. Clone repo and navigate to infra
cd packages/neuros-neurofm/infra

# 2. Set credentials
export COREWEAVE_API_KEY="your-api-key"
export COREWEAVE_ACCOUNT_ID="your-account-id"

# 3. Configure CoreWeave
cd coreweave
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your settings

# 4. Deploy cluster
make coreweave-init
make coreweave-apply

# 5. Configure kubectl
# Download kubeconfig from CoreWeave dashboard or use their CLI

# 6. Build and push Docker image
cd ..
export DOCKER_USERNAME="your-github-username"
export GITHUB_TOKEN="your-github-token"
make docker-build-push

# 7. Configure secrets
# Edit k8s/02-s3-secret.yaml with your S3 credentials
# Edit k8s/30-wandb-secret.yaml with your WandB API key
kubectl apply -f k8s/02-s3-secret.yaml
kubectl apply -f k8s/30-wandb-secret.yaml

# 8. Deploy Ray cluster
make k8s-apply

# 9. Check status
make k8s-status

# 10. Access Ray dashboard
make ray-dashboard
# Open http://localhost:8265

# 11. Start training
make train-small
```

### Option 2: Crusoe Cloud

```bash
# 1. Clone repo and navigate to infra
cd packages/neuros-neurofm/infra

# 2. Set credentials
export CRUSOE_API_TOKEN="your-api-token"

# 3. Configure Crusoe
cd crusoe
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your settings

# 4. Deploy cluster
make crusoe-init
make crusoe-apply

# 5. Get kubeconfig
make crusoe-kubeconfig
export KUBECONFIG=$PWD/kubeconfig.yaml

# 6-11. Same as CoreWeave steps 6-11
```

## ☁️ Cloud Provider Setup

### CoreWeave (Recommended)

#### Step 1: Get API Credentials

1. Sign up at https://cloud.coreweave.com/
2. Navigate to **Account Settings** → **API Access**
3. Create new API key
4. Note your **Account ID** and **API Key**

#### Step 2: Configure Terraform

```bash
cd coreweave
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
cluster_name       = "neurofmx-prod"
k8s_version        = "1.29"
region             = "ORD"  # Chicago - good H100 availability
gpu_machine_type   = "gpu.h100.80gb.hgx"
gpu_replicas       = 1  # Start with 1 node (8x H100)
```

**Available regions**:
- `ORD` - Chicago (recommended for H100 availability)
- `LAS` - Las Vegas
- `EWR` - New York/Newark
- `LGA` - New York/LaGuardia

#### Step 3: Deploy

```bash
# Set credentials (prefer environment variables)
export COREWEAVE_API_KEY="your-api-key"
export COREWEAVE_ACCOUNT_ID="your-account-id"

# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Deploy cluster
terraform apply -auto-approve
```

This creates:
- ✅ Managed Kubernetes cluster (CKS)
- ✅ 3x CPU nodes (system pool)
- ✅ 1x H100 HGX node (8 GPUs with NVLink)
- ✅ NVIDIA device plugins (pre-installed)
- ✅ CoreWeave CSI drivers (pre-installed)

#### Step 4: Get Kubeconfig

**Option A: CoreWeave Dashboard**
1. Go to https://cloud.coreweave.com/
2. Navigate to **Kubernetes** → Your cluster
3. Click **Download Kubeconfig**

**Option B: CoreWeave CLI**
```bash
# Install CoreWeave CLI
pip install coreweave-cli

# Download kubeconfig
cw kubectl config <cluster-name>
```

#### Step 5: Verify Access

```bash
kubectl get nodes
# Should show 3 CPU nodes + 1 H100 node

kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, gpus: .status.allocatable."nvidia.com/gpu"}'
```

### Crusoe Cloud (Alternative)

#### Step 1: Get API Token

1. Sign up at https://console.crusoecloud.com/
2. Navigate to **API Tokens**
3. Create new token with **Compute** permissions

#### Step 2: Setup SSH Key

```bash
# Generate SSH key if needed
ssh-keygen -t ed25519 -C "neurofmx-crusoe"

# Add to Crusoe console
# Go to Settings → SSH Keys → Add Key
```

#### Step 3: Configure Terraform

```bash
cd crusoe
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:

```hcl
cluster_name       = "neurofmx-crusoe"
region             = "us-northcentral1-a"
h100_instance_type = "gpu.h100-80gb.hgx.8x"
node_count         = 1
k3s_token          = "your-secure-random-token-here"  # Generate: openssl rand -base64 32
ssh_key_ids        = ["your-ssh-key-id-from-crusoe"]
```

#### Step 4: Deploy

```bash
export CRUSOE_API_TOKEN="your-api-token"

terraform init
terraform apply -auto-approve
```

This creates:
- ✅ H100 HGX instance(s)
- ✅ K3s Kubernetes (installed via cloud-init)
- ✅ NVIDIA drivers + container toolkit
- ✅ NCCL optimizations for NVLink

#### Step 5: Get Kubeconfig

```bash
# Terraform will output master IP
MASTER_IP=$(terraform output -raw master_public_ip)

# Download kubeconfig
scp root@${MASTER_IP}:/etc/rancher/k3s/k3s.yaml ./kubeconfig.yaml

# Fix server URL
sed -i "s/127.0.0.1/${MASTER_IP}/g" kubeconfig.yaml

# Use it
export KUBECONFIG=$PWD/kubeconfig.yaml
```

## 🐳 Docker Image Setup

### Step 1: Update Configuration

Edit `infra/Makefile`:

```makefile
DOCKER_REGISTRY := ghcr.io
DOCKER_USERNAME := your-github-username  # Change this!
```

Edit `infra/k8s/20-raycluster-neurofmx.yaml`:

```yaml
image: ghcr.io/your-github-username/neurofmx-train:latest  # Change this!
```

### Step 2: Build Image

```bash
cd infra
make docker-build
```

Or manually:

```bash
cd packages/neuros-neurofm
docker build -f docker/Dockerfile -t ghcr.io/your-username/neurofmx-train:latest .
```

### Step 3: Test Locally (Optional)

```bash
docker run --gpus all -it ghcr.io/your-username/neurofmx-train:latest

# Inside container
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
python -c "import mamba_ssm; print('Mamba OK')"
python -c "import flash_attn; print('FlashAttention OK')"
```

### Step 4: Push to Registry

**GitHub Container Registry**:

```bash
# Create personal access token with write:packages permission
# https://github.com/settings/tokens

export GITHUB_TOKEN="your-token"
echo $GITHUB_TOKEN | docker login ghcr.io -u your-username --password-stdin

make docker-push
```

**Docker Hub**:

```bash
docker login
docker tag ghcr.io/your-username/neurofmx-train:latest your-dockerhub-username/neurofmx-train:latest
docker push your-dockerhub-username/neurofmx-train:latest
```

## ☸️ Kubernetes Deployment

### Step 1: Configure Secrets

**S3 Storage** (`k8s/02-s3-secret.yaml`):

```yaml
stringData:
  AWS_ACCESS_KEY_ID: "your-access-key"
  AWS_SECRET_ACCESS_KEY: "your-secret-key"

  # For CoreWeave Object Storage:
  AWS_ENDPOINT_URL: "https://object.ord1.coreweave.com"

  # For AWS S3:
  # AWS_ENDPOINT_URL: "https://s3.amazonaws.com"

  AWS_REGION: "us-east-1"
  S3_BUCKET: "neurofmx-checkpoints"
```

**WandB** (`k8s/30-wandb-secret.yaml`):

```yaml
stringData:
  api-key: "your-wandb-api-key"  # Get from https://wandb.ai/authorize
```

### Step 2: Create Storage

**For CoreWeave**:

Edit `k8s/03-storage-pvc.yaml`:

```yaml
storageClassName: "ceph-filesystem"  # CoreWeave's RWX storage
```

**For Crusoe/K3s**:

You may need to setup NFS or use local-path:

```yaml
storageClassName: "local-path"  # K3s default
```

### Step 3: Deploy Resources

```bash
cd infra

# Apply secrets
kubectl apply -f k8s/02-s3-secret.yaml
kubectl apply -f k8s/30-wandb-secret.yaml

# Deploy everything
make k8s-apply
```

This deploys:
1. ✅ Namespace (`neurofmx`)
2. ✅ NVIDIA device plugin
3. ✅ Storage PVCs
4. ✅ KubeRay operator
5. ✅ Ray cluster (1 head + 8 workers)

### Step 4: Verify Deployment

```bash
# Check all resources
make k8s-status

# Should show:
# - 1 ray-head pod (Running)
# - 8 ray-worker pods (Running)
# - All with 1 GPU allocated per worker
```

### Step 5: Access Ray Dashboard

```bash
make ray-dashboard

# Open http://localhost:8265
```

You should see:
- 8 nodes (workers) with 1 GPU each
- Cluster resources: 8 GPUs total
- Connected and healthy

## 🎓 Training Jobs

### Download Data First

```bash
# SSH into Ray head pod
kubectl -n neurofmx exec -it svc/neurofmx-ray-head-svc -- /bin/bash

# Inside pod: Download all datasets
cd /workspace
bash scripts/download_all_cloud.sh \
    --data-dir /mnt/data \
    --parallel 6 \
    --modalities all

# This will download:
# - IBL spikes
# - Allen 2-Photon calcium
# - PhysioNet EEG
# - fMRI datasets
# - ECoG recordings
# - EMG data
# - LFP/iEEG data
```

### Run Training

**Small Model (Testing)**:

```bash
make train-small

# Or manually:
kubectl -n neurofmx exec -it svc/neurofmx-ray-head-svc -- \
    python /workspace/training/train_multimodal.py \
    --config /workspace/configs/model_small.yaml \
    --data_dir /mnt/data \
    --checkpoint_dir /mnt/checkpoints/small
```

**Medium Model (Recommended)**:

```bash
make train-medium
```

**Large Model (Full Scale)**:

```bash
make train-large
```

### Monitor Training

**Ray Dashboard**:
```bash
make ray-dashboard
# Open http://localhost:8265
# Go to "Jobs" tab to see active training
```

**WandB**:
- Open https://wandb.ai/
- Navigate to your project: `neurofmx`
- View real-time metrics, losses, GPU utilization

**Logs**:
```bash
# Ray head logs
make ray-logs

# Specific worker logs
kubectl -n neurofmx logs -f neurofmx-ray-worker-h100-workers-<pod-id>
```

## 📊 Monitoring & Debugging

### Check GPU Utilization

```bash
# From head pod
kubectl -n neurofmx exec -it svc/neurofmx-ray-head-svc -- nvidia-smi

# From specific worker
kubectl -n neurofmx exec -it neurofmx-ray-worker-h100-workers-<pod-id> -- nvidia-smi
```

### Check NCCL Communication

```bash
# Inside training, NCCL will log:
# - P2P access: OK (NVLink enabled)
# - IB access: Disabled
# - NET/Socket: Used for any non-P2P
```

### View Checkpoints

```bash
# List checkpoints in S3
kubectl -n neurofmx exec -it svc/neurofmx-ray-head-svc -- \
    aws s3 ls s3://neurofmx-checkpoints/ --endpoint-url=$AWS_ENDPOINT_URL
```

### Debug Pod Issues

```bash
# Describe pod
kubectl -n neurofmx describe pod neurofmx-ray-worker-h100-workers-<pod-id>

# Check events
kubectl -n neurofmx get events --sort-by='.lastTimestamp'

# Shell into worker
kubectl -n neurofmx exec -it neurofmx-ray-worker-h100-workers-<pod-id> -- /bin/bash
```

## 💰 Cost Optimization

### CoreWeave Pricing (Approximate)

- **H100 80GB HGX (8 GPUs)**: ~$24-30/hour on-demand
- **Storage**: ~$0.10/GB/month
- **Network**: Egress charges apply

**$500 Pilot Budget**:
- ~16-20 hours of H100 HGX time
- Focus on small→medium model training
- Use efficient data loading and checkpointing

### Tips to Reduce Costs

1. **Use Spot/Preemptible Instances** (if available):
   ```hcl
   preemptible = true  # In terraform.tfvars
   ```

2. **Scale Down When Not Training**:
   ```bash
   # Delete Ray cluster (keep data)
   kubectl -n neurofmx delete raycluster neurofmx-ray

   # Recreate when needed
   kubectl apply -f k8s/20-raycluster-neurofmx.yaml
   ```

3. **Optimize Batch Size**:
   - Use gradient accumulation to simulate larger batches
   - Maximize GPU utilization (check nvidia-smi)

4. **Checkpoint Frequently**:
   - Save every 1000-5000 steps
   - Can resume if preempted

5. **Use Mixed Precision**:
   - Already enabled in configs
   - Reduces memory and increases throughput

## 🔧 Troubleshooting

### Issue: Pods Stuck in Pending

**Check**:
```bash
kubectl -n neurofmx describe pod <pod-name>
```

**Common Causes**:
- No GPU nodes available → Check `kubectl get nodes`
- GPU already allocated → Check `kubectl describe node <node-name>`
- PVC not bound → Check `kubectl get pvc -n neurofmx`

**Fix**:
- Scale up GPU nodes in Terraform
- Delete stuck pods: `kubectl -n neurofmx delete pod <pod-name>`

### Issue: OOM (Out of Memory)

**Symptoms**:
- Pod killed with exit code 137
- CUDA out of memory errors

**Fix**:
1. Reduce batch size in config:
   ```yaml
   training:
     batch_size: 16  # Try 8 or 4
   ```

2. Increase gradient accumulation:
   ```yaml
   training:
     gradient_accumulation_steps: 4  # Try 8
   ```

3. Enable gradient checkpointing:
   ```yaml
   hardware:
     gradient_checkpointing: true
   ```

### Issue: Slow Data Loading

**Symptoms**:
- Low GPU utilization
- Training steps take long time

**Fix**:
1. Increase data workers:
   ```yaml
   data:
     num_workers: 8  # Try 16
   ```

2. Pin memory:
   ```yaml
   data:
     pin_memory: true
   ```

3. Use faster storage (local SSD if available)

### Issue: NCCL Timeout

**Symptoms**:
- NCCL watchdog timeout errors
- Distributed training hangs

**Fix**:
1. Check network connectivity between workers
2. Increase NCCL timeout:
   ```bash
   export NCCL_TIMEOUT=1800  # 30 minutes
   ```

3. Verify NVLink is enabled:
   ```bash
   nvidia-smi topo -m
   ```

### Issue: Can't Access Ray Dashboard

**Fix**:
```bash
# Check service
kubectl -n neurofmx get svc neurofmx-ray-dashboard

# Port-forward manually
kubectl -n neurofmx port-forward svc/neurofmx-ray-head-svc 8265:8265

# Or use LoadBalancer (costs extra)
# Edit k8s/20-raycluster-neurofmx.yaml:
spec:
  type: LoadBalancer
```

## 📚 Additional Resources

- **CoreWeave Docs**: https://docs.coreweave.com/
- **Crusoe Docs**: https://docs.crusoecloud.com/
- **KubeRay Docs**: https://docs.ray.io/en/latest/cluster/kubernetes/index.html
- **NeuroFMx Training Guide**: ../README_IMPLEMENTATION.md
- **Mamba SSM**: https://github.com/state-spaces/mamba
- **FlashAttention**: https://github.com/Dao-AILab/flash-attention

## 🆘 Support

For issues:
1. Check logs: `make ray-logs`
2. Check Ray dashboard: `make ray-dashboard`
3. Review K8s events: `kubectl -n neurofmx get events`
4. Open GitHub issue with logs

---

**Ready to train?** Start with: `make deploy-coreweave` or `make deploy-crusoe`
