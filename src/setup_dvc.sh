#!/bin/bash

# Simple DVC Setup Script
# Sets up DVC with S3 remote storage

set -e

echo "🚀 Setting up DVC..."

# Check if DVC is installed
if ! command -v dvc &> /dev/null; then
    echo "📦 Installing DVC..."
    pip install dvc dvc-s3
fi

# Initialize DVC if not done
if [ ! -d ".dvc" ]; then
    echo "🔧 Initializing DVC..."
    dvc init --no-scm
fi

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📋 Creating .env file..."
    cp .env.example .env
    echo "✅ Edit .env file with your AWS credentials: nano .env"
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Configure S3 remote if bucket is set
if [ ! -z "$S3_BUCKET_NAME" ] && [ "$S3_BUCKET_NAME" != "your-bucket-name" ]; then
    echo "🌐 Configuring S3 remote..."
    
    if ! dvc remote list | grep -q "s3remote"; then
        dvc remote add -d s3remote s3://$S3_BUCKET_NAME/dvc-storage
        dvc remote modify s3remote region ${AWS_REGION:-us-east-1}
    fi
    
    echo "✅ S3 remote configured"
else
    echo "⚠️  Set S3_BUCKET_NAME in .env file"
fi

# Remove credentials from DVC config for security
if [ -f ".dvc/config" ] && grep -q "access_key_id\|secret_access_key" .dvc/config; then
    echo "🔒 Removing credentials from DVC config..."
    sed -i.bak '/access_key_id/d; /secret_access_key/d' .dvc/config
    rm -f .dvc/config.bak
fi

echo "✅ DVC setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env: nano .env"
echo "2. Run pipeline: dvc repro"
echo "3. Push to S3: dvc push"