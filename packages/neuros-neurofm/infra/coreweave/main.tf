terraform {
  required_version = ">= 1.6"
  required_providers {
    coreweave = {
      source  = "coreweave/coreweave"
      version = "~> 0.5"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.33"
    }
  }
}

provider "coreweave" {
  # Set via ENV: COREWEAVE_API_KEY, COREWEAVE_ACCOUNT_ID
  # or use variables below (not recommended for commits).
  api_key    = var.coreweave_api_key
  account_id = var.coreweave_account_id
}

# Create a CKS Kubernetes cluster with H100 HGX-capable node group(s)
resource "coreweave_cks_cluster" "neurofmx" {
  name               = var.cluster_name
  kubernetes_version = var.k8s_version
  region             = var.region

  # System pool (small)
  node_pools = [
    {
      name         = "system-pool"
      node_class   = "cpu"
      replicas     = 3
      machine_type = "cpu.small"
    }
  ]

  # GPU pool (H100 HGX). Example machine_type may vary by region/stock; use CoreWeave docs/catalog.
  gpu_node_pools = [
    {
      name         = "hgx-h100-pool"
      gpu          = "H100"
      machine_type = var.gpu_machine_type # e.g., "gpu.h100.80gb.hgx"
      replicas     = var.gpu_replicas     # 1 node with 8x H100 (HGX)
      preemptible  = false
      labels = {
        gpu    = "h100"
        fabric = "nvlink"
      }
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "present"
        effect = "NoSchedule"
      }]
    }
  ]

  # Addon: CoreWeave-managed CSI + NVIDIA drivers are typically enabled by default on CKS.
  # Keep defaults or enable as needed per docs.
}

# Export kubeconfig for Kubernetes provider
data "coreweave_cks_cluster_kubeconfig" "neurofmx" {
  id = coreweave_cks_cluster.neurofmx.id
}

provider "kubernetes" {
  host                   = data.coreweave_cks_cluster_kubeconfig.neurofmx.host
  cluster_ca_certificate = base64decode(data.coreweave_cks_cluster_kubeconfig.neurofmx.cluster_ca_certificate)
  token                  = data.coreweave_cks_cluster_kubeconfig.neurofmx.token
}
