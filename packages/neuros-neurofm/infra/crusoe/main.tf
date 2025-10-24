terraform {
  required_version = ">= 1.6"
  required_providers {
    crusoe = {
      source  = "crusoecloud/crusoe"
      version = "~> 0.5"
    }
  }
}

provider "crusoe" {
  # Set via env: CRUSOE_API_TOKEN
  token = var.crusoe_api_token
}

variable "crusoe_api_token" {
  type        = string
  sensitive   = true
  default     = null
  description = "Crusoe API token (or set CRUSOE_API_TOKEN env)"
}

variable "region" {
  type        = string
  default     = "us-northcentral1-a"
  description = "Crusoe region"
}

variable "h100_instance_type" {
  type        = string
  default     = "gpu.h100-80gb.hgx.8x"
  description = "H100 HGX instance type (8x H100 80GB with NVLink/NVSwitch)"
}

variable "node_count" {
  type        = number
  default     = 1
  description = "Number of H100 HGX nodes"
}

variable "cluster_name" {
  type        = string
  default     = "neurofmx-crusoe"
  description = "Cluster name"
}

# Create a security group (SSH + K8s control plane ports)
resource "crusoe_security_group" "k3s" {
  name        = "${var.cluster_name}-sg"
  description = "Security group for K3s cluster nodes"
  region      = var.region

  ingress {
    protocol    = "tcp"
    port_range  = "22"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # K8s API server
  ingress {
    protocol    = "tcp"
    port_range  = "6443"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Ray dashboard
  ingress {
    protocol    = "tcp"
    port_range  = "8265"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow all outbound
  egress {
    protocol    = "-1"
    port_range  = "0-65535"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Provision H100 HGX instances and install K3s via cloud-init
resource "crusoe_instance" "h100_nodes" {
  count              = var.node_count
  name               = "${var.cluster_name}-h100-${count.index}"
  region             = var.region
  type               = var.h100_instance_type
  image              = "ubuntu-22.04"
  ssh_key_ids        = var.ssh_key_ids
  security_group_ids = [crusoe_security_group.k3s.id]

  user_data = templatefile("${path.module}/cloudinit-k3s.yaml", {
    node_index  = count.index
    is_master   = count.index == 0
    master_ip   = count.index == 0 ? "" : crusoe_instance.h100_nodes[0].public_ip
    k3s_token   = var.k3s_token
  })
}

variable "ssh_key_ids" {
  type        = list(string)
  default     = []
  description = "SSH key IDs for instances"
}

variable "k3s_token" {
  type        = string
  sensitive   = true
  default     = "neurofmx-default-token"
  description = "K3s cluster token for node joining"
}

output "master_public_ip" {
  description = "K3s master node public IP"
  value       = crusoe_instance.h100_nodes[0].public_ip
}

output "node_ips" {
  description = "All node public IPs"
  value       = [for n in crusoe_instance.h100_nodes : n.public_ip]
}

output "kubeconfig_command" {
  description = "Command to fetch kubeconfig"
  value       = "scp root@${crusoe_instance.h100_nodes[0].public_ip}:/etc/rancher/k3s/k3s.yaml ./kubeconfig.yaml"
}
