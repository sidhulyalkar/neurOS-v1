#!/bin/bash
# AWS Setup Script for NeuroFM-X Training
#
# This script sets up an AWS A100 instance for training
#
# Usage:
#   ./aws_setup.sh <instance-id>

set -e

INSTANCE_ID=$1
REGION="us-east-1"
S3_BUCKET="neurofm-training"  # Change to your bucket

if [ -z "$INSTANCE_ID" ]; then
    echo "Usage: $0 <instance-id>"
    exit 1
fi

echo "========================================="
echo "AWS NeuroFM-X Training Setup"
echo "========================================="
echo "Instance: $INSTANCE_ID"
echo "Region: $REGION"
echo "S3 Bucket: $S3_BUCKET"
echo ""

# 1. Create S3 bucket if it doesn't exist
echo "[1/6] Creating S3 bucket..."
aws s3 mb s3://$S3_BUCKET --region $REGION 2>/dev/null || echo "Bucket already exists"

# 2. Upload data to S3
echo "[2/6] Uploading data to S3..."
aws s3 sync ./data s3://$S3_BUCKET/data --region $REGION

# 3. Upload code to S3
echo "[3/6] Uploading code to S3..."
tar -czf neurofm_code.tar.gz \
    src/ \
    training/ \
    scripts/ \
    configs/ \
    deployment/ \
    requirements.txt \
    setup.py
aws s3 cp neurofm_code.tar.gz s3://$S3_BUCKET/code/ --region $REGION
rm neurofm_code.tar.gz

# 4. SSH into instance and setup
echo "[4/6] Connecting to instance and setting up..."
ssh -i ~/.ssh/aws_key.pem ec2-user@$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text) << 'ENDSSH'

# Update system
sudo yum update -y

# Install Docker
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
  sudo tee /etc/yum.repos.d/nvidia-docker.repo
sudo yum install -y nvidia-docker2
sudo systemctl restart docker

# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# Download code from S3
aws s3 cp s3://neurofm-training/code/neurofm_code.tar.gz .
tar -xzf neurofm_code.tar.gz
rm neurofm_code.tar.gz

# Download data from S3
mkdir -p data
aws s3 sync s3://neurofm-training/data ./data

echo "Setup complete!"
ENDSSH

# 5. Build Docker image
echo "[5/6] Building Docker image on instance..."
ssh -i ~/.ssh/aws_key.pem ec2-user@$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text) << 'ENDSSH'

cd neurofm
docker build -t neurofm:latest -f deployment/Dockerfile .
ENDSSH

# 6. Start training
echo "[6/6] Starting training..."
echo "To start training, SSH into the instance and run:"
echo "  docker run --gpus all -v /home/ec2-user/data:/data neurofm:latest --config configs/cloud_aws_a100.yaml"

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To monitor training:"
echo "  ssh -i ~/.ssh/aws_key.pem ec2-user@INSTANCE_IP"
echo "  docker logs -f <container-id>"
echo ""
echo "To download checkpoints:"
echo "  aws s3 sync s3://$S3_BUCKET/checkpoints ./checkpoints"
