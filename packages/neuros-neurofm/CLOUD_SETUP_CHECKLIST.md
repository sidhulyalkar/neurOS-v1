# ☁️ Your Cloud Setup Checklist

This is your personal checklist for deploying NeuroFMx to the cloud. Follow these steps in order.

## ✅ Pre-Deployment Checklist

### 1. Cloud Provider Account Setup

**Choose your provider** (check one):
- [ ] **CoreWeave** (Recommended - fastest setup with managed Kubernetes)
  - [ ] Create account at https://cloud.coreweave.com/
  - [ ] Get API Key from Account Settings → API Access
  - [ ] Note your Account ID
  - [ ] Set environment variables:
    ```bash
    export COREWEAVE_API_KEY="your-api-key"
    export COREWEAVE_ACCOUNT_ID="your-account-id"
    ```

- [ ] **Crusoe Cloud** (Alternative - more control, self-managed K3s)
  - [ ] Create account at https://console.crusoecloud.com/
  - [ ] Generate API token with Compute permissions
  - [ ] Create SSH key and add to Crusoe console
  - [ ] Set environment variable:
    ```bash
    export CRUSOE_API_TOKEN="your-api-token"
    ```

### 2. Local Tools Installation

- [ ] Install Terraform (>= 1.6):
  ```bash
  # macOS
  brew install terraform

  # Linux
  wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
  unzip terraform_1.6.0_linux_amd64.zip
  sudo mv terraform /usr/local/bin/

  # Verify
  terraform version
  ```

- [ ] Install kubectl:
  ```bash
  # macOS
  brew install kubectl

  # Linux
  curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  chmod +x kubectl
  sudo mv kubectl /usr/local/bin/

  # Verify
  kubectl version --client
  ```

- [ ] Install Docker:
  - [ ] Follow https://docs.docker.com/get-docker/
  - [ ] Verify: `docker --version`

- [ ] Install Make (usually pre-installed on Linux/macOS):
  ```bash
  make --version
  ```

### 3. Container Registry Setup

**Choose your registry** (check one):

- [ ] **GitHub Container Registry (ghcr.io)** - Recommended for public repos
  - [ ] Create Personal Access Token: https://github.com/settings/tokens
  - [ ] Grant `write:packages` permission
  - [ ] Save token securely
  - [ ] Your username: `___________________`
  - [ ] Login test:
    ```bash
    export GITHUB_TOKEN="your-token"
    echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin
    ```

- [ ] **Docker Hub**
  - [ ] Create account at https://hub.docker.com/
  - [ ] Login: `docker login`

### 4. S3 Storage Setup

**For checkpoints and data:**

- [ ] **CoreWeave Object Storage** (if using CoreWeave):
  - [ ] Create bucket in CoreWeave dashboard
  - [ ] Get S3 credentials (Access Key + Secret Key)
  - [ ] Endpoint: `https://object.ord1.coreweave.com` (adjust for your region)

- [ ] **AWS S3** (alternative):
  - [ ] Create S3 bucket: `neurofmx-checkpoints`
  - [ ] Create IAM user with S3 access
  - [ ] Get Access Key + Secret Key

- [ ] **Your S3 Credentials**:
  ```
  Access Key ID: ___________________
  Secret Access Key: ___________________
  Endpoint URL: ___________________
  Bucket Name: ___________________
  ```

### 5. Experiment Tracking Setup

- [ ] **WandB** (Weights & Biases):
  - [ ] Create account at https://wandb.ai/
  - [ ] Get API key: https://wandb.ai/authorize
  - [ ] Your API key: `___________________`

## 🚀 Deployment Steps

### Phase 1: Infrastructure (15-20 minutes)

- [ ] **1. Clone and navigate to infra**:
  ```bash
  cd packages/neuros-neurofm/infra
  ```

- [ ] **2. Configure Terraform**:

  **If using CoreWeave**:
  ```bash
  cd coreweave
  cp terraform.tfvars.example terraform.tfvars
  # Edit terraform.tfvars with your settings
  nano terraform.tfvars  # or use any editor
  ```

  **If using Crusoe**:
  ```bash
  cd crusoe
  cp terraform.tfvars.example terraform.tfvars
  # Edit terraform.tfvars with your settings
  nano terraform.tfvars
  ```

- [ ] **3. Deploy cluster**:

  **CoreWeave**:
  ```bash
  make coreweave-init
  make coreweave-apply
  # Wait ~10-15 minutes for cluster to be ready
  ```

  **Crusoe**:
  ```bash
  make crusoe-init
  make crusoe-apply
  # Wait ~10-15 minutes for instances + K3s setup
  ```

- [ ] **4. Configure kubectl**:

  **CoreWeave**:
  - Download kubeconfig from CoreWeave dashboard
  - Or use CoreWeave CLI: `cw kubectl config <cluster-name>`

  **Crusoe**:
  ```bash
  make crusoe-kubeconfig
  export KUBECONFIG=$PWD/kubeconfig.yaml
  ```

- [ ] **5. Verify cluster**:
  ```bash
  kubectl get nodes
  # Should see CPU nodes + H100 GPU node(s)
  ```

### Phase 2: Container Image (10-15 minutes)

- [ ] **6. Update configuration files**:

  Edit `infra/Makefile`:
  ```makefile
  DOCKER_USERNAME := YOUR_GITHUB_USERNAME  # Line 4
  ```

  Edit `infra/k8s/20-raycluster-neurofmx.yaml`:
  ```yaml
  image: ghcr.io/YOUR_USERNAME/neurofmx-train:latest  # Lines 518, 578
  ```

- [ ] **7. Build Docker image**:
  ```bash
  cd infra
  make docker-build
  # Wait ~10-15 minutes for build
  ```

- [ ] **8. Test image locally (optional)**:
  ```bash
  docker run --gpus all -it ghcr.io/YOUR_USERNAME/neurofmx-train:latest
  # Test: python -c "import torch; print(torch.cuda.device_count())"
  # Exit: exit
  ```

- [ ] **9. Push to registry**:
  ```bash
  export GITHUB_TOKEN="your-token"
  make docker-push
  ```

### Phase 3: Kubernetes Deployment (10 minutes)

- [ ] **10. Configure secrets**:

  **S3 Secret** (`k8s/02-s3-secret.yaml`):
  ```bash
  nano k8s/02-s3-secret.yaml
  # Fill in:
  #   AWS_ACCESS_KEY_ID
  #   AWS_SECRET_ACCESS_KEY
  #   AWS_ENDPOINT_URL
  #   S3_BUCKET
  ```

  **WandB Secret** (`k8s/30-wandb-secret.yaml`):
  ```bash
  nano k8s/30-wandb-secret.yaml
  # Fill in your WandB API key
  ```

  **Apply secrets**:
  ```bash
  kubectl apply -f k8s/02-s3-secret.yaml
  kubectl apply -f k8s/30-wandb-secret.yaml
  ```

- [ ] **11. Update storage class** (if needed):

  Edit `k8s/03-storage-pvc.yaml`:
  ```yaml
  storageClassName: "ceph-filesystem"  # CoreWeave
  # OR
  storageClassName: "local-path"       # Crusoe/K3s
  ```

- [ ] **12. Deploy Ray cluster**:
  ```bash
  make k8s-apply
  # Wait ~5 minutes for pods to start
  ```

- [ ] **13. Verify deployment**:
  ```bash
  make k8s-status
  # Should show:
  #   1 ray-head pod (Running)
  #   8 ray-worker pods (Running)
  #   All with GPUs allocated
  ```

### Phase 4: Data Download (1-3 hours, runs in background)

- [ ] **14. SSH into Ray head**:
  ```bash
  kubectl -n neurofmx exec -it svc/neurofmx-ray-head-svc -- /bin/bash
  ```

- [ ] **15. Download datasets**:
  ```bash
  cd /workspace
  bash scripts/download_all_cloud.sh \
      --data-dir /mnt/data \
      --parallel 6 \
      --modalities all

  # This runs in background. You can monitor with:
  tail -f logs/downloads/*.log
  ```

- [ ] **16. Verify data**:
  ```bash
  ls -lh /mnt/data/*/processed/train/ | head
  cat /mnt/data/data_manifest.txt
  ```

### Phase 5: Training (Start your run!)

- [ ] **17. Access Ray dashboard**:
  ```bash
  # In a new terminal:
  cd packages/neuros-neurofm/infra
  make ray-dashboard
  # Open http://localhost:8265
  ```

- [ ] **18. Start training**:

  **Test run (small model, 30 min)**:
  ```bash
  make train-small
  ```

  **Full run (medium model, several hours)**:
  ```bash
  make train-medium
  ```

- [ ] **19. Monitor training**:
  - [ ] Ray Dashboard: http://localhost:8265 (Jobs tab)
  - [ ] WandB: https://wandb.ai/ (your project)
  - [ ] Logs: `make ray-logs`

## 📊 Monitoring Checklist

- [ ] **GPU utilization**:
  ```bash
  kubectl -n neurofmx exec -it svc/neurofmx-ray-head-svc -- nvidia-smi
  # Should show ~90%+ GPU utilization when training
  ```

- [ ] **Checkpoints saving**:
  ```bash
  kubectl -n neurofmx exec -it svc/neurofmx-ray-head-svc -- \
      aws s3 ls s3://your-bucket/ --endpoint-url=$AWS_ENDPOINT_URL
  ```

- [ ] **WandB logging**:
  - Open WandB dashboard
  - Check for real-time metrics

## 💰 Cost Management Checklist

- [ ] **Set budget alert** in cloud provider dashboard

- [ ] **Scale down when not training**:
  ```bash
  # Delete Ray cluster (keeps data)
  kubectl -n neurofmx delete raycluster neurofmx-ray

  # Recreate later
  kubectl apply -f k8s/20-raycluster-neurofmx.yaml
  ```

- [ ] **Tear down fully when done**:
  ```bash
  make k8s-delete
  make coreweave-destroy  # or make crusoe-destroy
  ```

## 🎯 Your $500 Pilot Plan

Based on CoreWeave H100 HGX pricing (~$24-30/hour):

- [ ] **Budget**: $500 → ~16-20 hours of compute
- [ ] **Week 1** (4 hours, $100):
  - [ ] Setup and verification (1 hour)
  - [ ] Small model test runs (1 hour)
  - [ ] Data quality checks (2 hours)

- [ ] **Week 2** (8 hours, $200):
  - [ ] Medium model training (8 hours)
  - [ ] Hyperparameter tuning

- [ ] **Week 3** (4-6 hours, $100-150):
  - [ ] Extended medium model training
  - [ ] Multi-modality experiments

- [ ] **Buffer** ($50):
  - [ ] Debugging, retries, extended runs

## 🔥 Quick Commands Reference

```bash
# Status
make k8s-status

# Dashboard
make ray-dashboard

# Logs
make ray-logs

# Train
make train-small
make train-medium
make train-large

# Scale down
kubectl -n neurofmx delete raycluster neurofmx-ray

# Tear down
make k8s-delete
make coreweave-destroy  # or crusoe-destroy
```

## ✅ Success Criteria

You know it's working when:
- [ ] Ray dashboard shows 8 workers with 1 GPU each
- [ ] GPU utilization is 80%+ during training
- [ ] WandB shows decreasing loss curves
- [ ] Checkpoints are saving to S3 every 1000-5000 steps
- [ ] Training is stable (no OOM or NCCL errors)

---

**Ready?** Start with Phase 1! 🚀

**Need help?** Check `infra/README.md` for detailed troubleshooting.
