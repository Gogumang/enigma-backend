# ============================================
# ECR Repository
# ============================================

resource "aws_ecr_repository" "enigma_api" {
  name                 = "enigma-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name        = "${var.project_name}-ecr"
    Environment = var.environment
    Project     = var.project_name
  }
}

# 오래된 이미지 자동 삭제 정책 (최근 5개만 유지)
resource "aws_ecr_lifecycle_policy" "enigma_api" {
  repository = aws_ecr_repository.enigma_api.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = {
        type = "expire"
      }
    }]
  })
}
