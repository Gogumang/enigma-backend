# EC2 Instance
resource "aws_instance" "enigma" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  subnet_id              = aws_subnet.public.id
  vpc_security_group_ids = [aws_security_group.enigma.id]

  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true

    tags = {
      Name = "${var.project_name}-root-volume"
    }
  }

  user_data = <<-EOF
    #!/bin/bash
    set -e

    # Update system
    apt-get update
    apt-get upgrade -y

    # Install AWS CLI
    apt-get install -y awscli

    # Install Docker
    apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

    # Add ubuntu user to docker group
    usermod -aG docker ubuntu

    # Setup swap (4GB)
    fallocate -l 4G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab

    # Optimize for ML workloads
    echo 'vm.swappiness=10' >> /etc/sysctl.conf
    sysctl -p

    # Create app directory
    mkdir -p /home/ubuntu/app
    chown ubuntu:ubuntu /home/ubuntu/app

    # Signal completion
    touch /home/ubuntu/.setup-complete
  EOF

  tags = {
    Name = "${var.project_name}-server"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Elastic IP
resource "aws_eip" "enigma" {
  domain = "vpc"

  tags = {
    Name = "${var.project_name}-eip"
  }
}

# Elastic IP Association
resource "aws_eip_association" "enigma" {
  instance_id   = aws_instance.enigma.id
  allocation_id = aws_eip.enigma.id
}
