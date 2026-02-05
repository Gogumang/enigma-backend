output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.enigma.id
}

output "elastic_ip" {
  description = "Elastic IP address"
  value       = aws_eip.enigma.public_ip
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${aws_eip.enigma.public_ip}"
}

output "api_url" {
  description = "API URL (HTTP)"
  value       = "http://${aws_eip.enigma.public_ip}/api"
}

output "health_check_url" {
  description = "Health check URL"
  value       = "http://${aws_eip.enigma.public_ip}/api/health"
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.enigma.id
}

output "subnet_id" {
  description = "Public subnet ID"
  value       = aws_subnet.public.id
}

# ============================================
# ECR Outputs
# ============================================

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.enigma_api.repository_url
}

output "ecr_repository_name" {
  description = "ECR repository name"
  value       = aws_ecr_repository.enigma_api.name
}

output "ecr_registry_id" {
  description = "ECR registry ID (AWS Account ID)"
  value       = aws_ecr_repository.enigma_api.registry_id
}
