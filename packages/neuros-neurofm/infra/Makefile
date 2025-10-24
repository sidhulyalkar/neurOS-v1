SHELL := /bin/bash
KUBE := kubectl
K8S_DIR := ./k8s
DOCKER_REGISTRY := ghcr.io
DOCKER_USERNAME := YOUR_USERNAME  # Change this!
IMAGE_NAME := neurofmx-train
IMAGE_TAG := latest

.PHONY: help
help:
	@echo "NeuroFMx Cloud Deployment Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  CoreWeave:"
	@echo "    coreweave-init      - Initialize Terraform for CoreWeave"
	@echo "    coreweave-plan      - Plan CoreWeave infrastructure"
	@echo "    coreweave-apply     - Deploy CoreWeave cluster"
	@echo "    coreweave-destroy   - Tear down CoreWeave cluster"
	@echo "    coreweave-kubeconfig- Get kubeconfig for CoreWeave"
	@echo ""
	@echo "  Crusoe:"
	@echo "    crusoe-init         - Initialize Terraform for Crusoe"
	@echo "    crusoe-plan         - Plan Crusoe infrastructure"
	@echo "    crusoe-apply        - Deploy Crusoe cluster"
	@echo "    crusoe-destroy      - Tear down Crusoe cluster"
	@echo "    crusoe-kubeconfig   - Get kubeconfig from Crusoe master"
	@echo ""
	@echo "  Kubernetes:"
	@echo "    k8s-apply           - Deploy all K8s resources (Ray cluster)"
	@echo "    k8s-delete          - Delete all K8s resources"
	@echo "    k8s-status          - Check cluster status"
	@echo "    ray-dashboard       - Port-forward Ray dashboard (localhost:8265)"
	@echo "    ray-logs            - Tail Ray head logs"
	@echo ""
	@echo "  Docker:"
	@echo "    docker-build        - Build training container image"
	@echo "    docker-push         - Push image to registry"
	@echo "    docker-build-push   - Build and push in one command"
	@echo ""
	@echo "  Training:"
	@echo "    train-small         - Submit small model training job"
	@echo "    train-medium        - Submit medium model training job"
	@echo "    train-large         - Submit large model training job"
	@echo ""
	@echo "  Complete workflows:"
	@echo "    deploy-coreweave    - Full CoreWeave deployment"
	@echo "    deploy-crusoe       - Full Crusoe deployment"

# ═══════════════════════════════════════════════════════════════
# CoreWeave
# ═══════════════════════════════════════════════════════════════

.PHONY: coreweave-init
coreweave-init:
	@echo "Initializing Terraform for CoreWeave..."
	cd coreweave && terraform init

.PHONY: coreweave-plan
coreweave-plan:
	@echo "Planning CoreWeave infrastructure..."
	cd coreweave && terraform plan

.PHONY: coreweave-apply
coreweave-apply:
	@echo "Deploying CoreWeave CKS cluster..."
	cd coreweave && terraform apply -auto-approve
	@echo "Waiting for cluster to be ready..."
	sleep 30

.PHONY: coreweave-destroy
coreweave-destroy:
	@echo "WARNING: This will destroy your CoreWeave cluster!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		cd coreweave && terraform destroy -auto-approve; \
	fi

.PHONY: coreweave-kubeconfig
coreweave-kubeconfig:
	@echo "Fetching kubeconfig from CoreWeave..."
	cd coreweave && terraform output -raw kubeconfig_host

# ═══════════════════════════════════════════════════════════════
# Crusoe
# ═══════════════════════════════════════════════════════════════

.PHONY: crusoe-init
crusoe-init:
	@echo "Initializing Terraform for Crusoe..."
	cd crusoe && terraform init

.PHONY: crusoe-plan
crusoe-plan:
	@echo "Planning Crusoe infrastructure..."
	cd crusoe && terraform plan

.PHONY: crusoe-apply
crusoe-apply:
	@echo "Deploying Crusoe H100 instances with K3s..."
	cd crusoe && terraform apply -auto-approve
	@echo "Waiting for K3s to initialize..."
	sleep 120

.PHONY: crusoe-destroy
crusoe-destroy:
	@echo "WARNING: This will destroy your Crusoe cluster!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		cd crusoe && terraform destroy -auto-approve; \
	fi

.PHONY: crusoe-kubeconfig
crusoe-kubeconfig:
	@echo "Fetching kubeconfig from Crusoe master..."
	@MASTER_IP=$$(cd crusoe && terraform output -raw master_public_ip); \
	scp root@$$MASTER_IP:/etc/rancher/k3s/k3s.yaml ./kubeconfig.yaml && \
	sed -i "s/127.0.0.1/$$MASTER_IP/g" ./kubeconfig.yaml && \
	echo "Kubeconfig saved to ./kubeconfig.yaml" && \
	echo "Export with: export KUBECONFIG=$$PWD/kubeconfig.yaml"

# ═══════════════════════════════════════════════════════════════
# Kubernetes
# ═══════════════════════════════════════════════════════════════

.PHONY: k8s-apply
k8s-apply:
	@echo "Deploying Kubernetes resources..."
	$(KUBE) apply -f $(K8S_DIR)/00-namespace.yaml
	@echo "Waiting for namespace..."
	@sleep 5
	$(KUBE) apply -f $(K8S_DIR)/01-nvidia-device-plugin.yaml || true
	@echo "Edit and apply secrets with your credentials:"
	@echo "  kubectl apply -f $(K8S_DIR)/02-s3-secret.yaml"
	@echo "  kubectl apply -f $(K8S_DIR)/30-wandb-secret.yaml"
	@read -p "Press enter when secrets are configured..."
	$(KUBE) apply -f $(K8S_DIR)/03-storage-pvc.yaml
	$(KUBE) apply -f $(K8S_DIR)/10-kuberay-operator.yaml
	@echo "Waiting for KubeRay operator..."
	@sleep 30
	$(KUBE) apply -f $(K8S_DIR)/20-raycluster-neurofmx.yaml
	@echo "Waiting for Ray cluster to start..."
	@sleep 10
	@$(MAKE) k8s-status

.PHONY: k8s-delete
k8s-delete:
	@echo "Deleting Kubernetes resources..."
	-$(KUBE) delete -f $(K8S_DIR)/20-raycluster-neurofmx.yaml
	@sleep 10
	-$(KUBE) delete -f $(K8S_DIR)/10-kuberay-operator.yaml
	-$(KUBE) delete -f $(K8S_DIR)/03-storage-pvc.yaml
	-$(KUBE) delete -f $(K8S_DIR)/30-wandb-secret.yaml
	-$(KUBE) delete -f $(K8S_DIR)/02-s3-secret.yaml
	-$(KUBE) delete -f $(K8S_DIR)/01-nvidia-device-plugin.yaml
	-$(KUBE) delete -f $(K8S_DIR)/00-namespace.yaml

.PHONY: k8s-status
k8s-status:
	@echo "═══════════════════════════════════════════════════════"
	@echo "Cluster Status"
	@echo "═══════════════════════════════════════════════════════"
	@echo ""
	@echo "Nodes:"
	$(KUBE) get nodes -o wide
	@echo ""
	@echo "NeuroFMx Pods:"
	$(KUBE) -n neurofmx get pods -o wide
	@echo ""
	@echo "Ray System:"
	$(KUBE) -n ray-system get pods
	@echo ""
	@echo "PVCs:"
	$(KUBE) -n neurofmx get pvc

.PHONY: ray-dashboard
ray-dashboard:
	@echo "Port-forwarding Ray dashboard to localhost:8265..."
	@echo "Open http://localhost:8265 in your browser"
	$(KUBE) -n neurofmx port-forward svc/neurofmx-ray-dashboard 8265:8265

.PHONY: ray-logs
ray-logs:
	@echo "Tailing Ray head logs..."
	$(KUBE) -n neurofmx logs -f -l role=ray-head

# ═══════════════════════════════════════════════════════════════
# Docker
# ═══════════════════════════════════════════════════════════════

.PHONY: docker-build
docker-build:
	@echo "Building Docker image..."
	cd .. && docker build -f docker/Dockerfile -t $(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(IMAGE_NAME):$(IMAGE_TAG) .

.PHONY: docker-push
docker-push:
	@echo "Pushing Docker image to $(DOCKER_REGISTRY)..."
	docker push $(DOCKER_REGISTRY)/$(DOCKER_USERNAME)/$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: docker-build-push
docker-build-push: docker-build docker-push

# ═══════════════════════════════════════════════════════════════
# Training Jobs
# ═══════════════════════════════════════════════════════════════

.PHONY: train-small
train-small:
	@echo "Submitting small model training job..."
	$(KUBE) -n neurofmx exec -it svc/neurofmx-ray-head-svc -- \
		python /workspace/training/train_multimodal.py \
		--config /workspace/configs/model_small.yaml \
		--data_dir /mnt/data

.PHONY: train-medium
train-medium:
	@echo "Submitting medium model training job..."
	$(KUBE) -n neurofmx exec -it svc/neurofmx-ray-head-svc -- \
		python /workspace/training/train_multimodal.py \
		--config /workspace/configs/model_medium.yaml \
		--data_dir /mnt/data

.PHONY: train-large
train-large:
	@echo "Submitting large model training job..."
	$(KUBE) -n neurofmx exec -it svc/neurofmx-ray-head-svc -- \
		python /workspace/training/train_multimodal.py \
		--config /workspace/configs/model_large.yaml \
		--data_dir /mnt/data

# ═══════════════════════════════════════════════════════════════
# Complete Workflows
# ═══════════════════════════════════════════════════════════════

.PHONY: deploy-coreweave
deploy-coreweave: coreweave-init coreweave-apply
	@echo ""
	@echo "═══════════════════════════════════════════════════════"
	@echo "CoreWeave cluster deployed!"
	@echo "═══════════════════════════════════════════════════════"
	@echo ""
	@echo "Next steps:"
	@echo "1. Configure kubectl to use CoreWeave cluster"
	@echo "2. Edit secrets: $(K8S_DIR)/02-s3-secret.yaml and $(K8S_DIR)/30-wandb-secret.yaml"
	@echo "3. Deploy K8s resources: make k8s-apply"
	@echo "4. Build and push Docker image: make docker-build-push"
	@echo "5. Start training: make train-small"

.PHONY: deploy-crusoe
deploy-crusoe: crusoe-init crusoe-apply crusoe-kubeconfig
	@echo ""
	@echo "═══════════════════════════════════════════════════════"
	@echo "Crusoe cluster deployed!"
	@echo "═══════════════════════════════════════════════════════"
	@echo ""
	@echo "Next steps:"
	@echo "1. Export kubeconfig: export KUBECONFIG=$$PWD/kubeconfig.yaml"
	@echo "2. Edit secrets: $(K8S_DIR)/02-s3-secret.yaml and $(K8S_DIR)/30-wandb-secret.yaml"
	@echo "3. Deploy K8s resources: make k8s-apply"
	@echo "4. Build and push Docker image: make docker-build-push"
	@echo "5. Start training: make train-small"
