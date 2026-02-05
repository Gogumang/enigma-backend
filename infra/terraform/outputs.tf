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
