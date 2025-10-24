output "cluster_id" {
  description = "CKS cluster ID"
  value       = coreweave_cks_cluster.neurofmx.id
}

output "kubeconfig_host" {
  description = "Kubernetes API server endpoint"
  value       = data.coreweave_cks_cluster_kubeconfig.neurofmx.host
}

output "cluster_name" {
  description = "Cluster name"
  value       = var.cluster_name
}

output "region" {
  description = "Cluster region"
  value       = var.region
}
