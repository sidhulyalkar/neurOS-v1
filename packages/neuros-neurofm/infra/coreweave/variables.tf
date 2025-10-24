variable "coreweave_api_key" {
  type        = string
  sensitive   = true
  description = "CoreWeave API key (or set COREWEAVE_API_KEY env)"
  default     = null
}

variable "coreweave_account_id" {
  type        = string
  sensitive   = true
  description = "CoreWeave account id (or set COREWEAVE_ACCOUNT_ID env)"
  default     = null
}

variable "cluster_name" {
  type        = string
  default     = "neurofmx-cks"
  description = "Name of the CKS cluster"
}

variable "k8s_version" {
  type        = string
  default     = "1.29"
  description = "Kubernetes version"
}

variable "region" {
  type        = string
  default     = "ORD"
  description = "CoreWeave region (ORD, LAS, EWR, etc.)"
}

variable "gpu_machine_type" {
  type        = string
  default     = "gpu.h100.80gb.hgx"
  description = "H100 HGX machine type"
}

variable "gpu_replicas" {
  type        = number
  default     = 1
  description = "Number of HGX nodes (1 node = 8x H100 GPUs)"
}
