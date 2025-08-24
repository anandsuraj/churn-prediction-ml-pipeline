#!/bin/bash

# DVC Credentials Setup Script
# This script helps set up DVC with secure credential management

echo "🔧 Setting up DVC with secure credentials..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📋 Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env file. Please edit it with your credentials."
    echo "📝 Edit .env file: nano .env"
else
    echo "✅ .env file already exists"
fi

# Check if DVC is initialized
if [ ! -d ".dvc" ]; then
    echo "🚀 Initializing DVC..."
    dvc init
    echo "✅ DVC initialized"
else
    echo "✅ DVC already initialized"
fi

# Load environment variables
if [ -f ".env" ]; then
    echo "📥 Loading environment variables..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Configure DVC remote if S3 bucket is specified
if [ ! -z "$S3_BUCKET_NAME" ]; then
    echo "🌐 Configuring S3 remote storage..."
    
    # Check if remote already exists
    if dvc remote list | grep -q "s3remote"; then
        echo "✅ S3 remote already configured"
    else
        dvc remote add -d s3remote s3://$S3_BUCKET_NAME/dvc-storage
        dvc remote modify s3remote region ${AWS_REGION:-us-east-1}
        echo "✅ S3 remote configured"
    fi
    
    # Test connection
    echo "🔍 Testing S3 connection..."
    if dvc remote modify s3remote --test; then
        echo "✅ S3 connection successful"
    else
        echo "❌ S3 connection failed. Please check your credentials in .env file"
    fi
else
    echo "⚠️  S3_BUCKET_NAME not set in .env file"
fi

# Check if credentials are in DVC config (security issue)
if grep -q "access_key_id\|secret_access_key" .dvc/config 2>/dev/null; then
    echo "⚠️  WARNING: Credentials found in .dvc/config file!"
    echo "🔒 Removing credentials from DVC config for security..."
    
    # Remove credentials from config
    sed -i.bak '/access_key_id/d; /secret_access_key/d' .dvc/config
    rm -f .dvc/config.bak
    
    echo "✅ Credentials removed from DVC config"
    echo "🔐 Credentials will be read from environment variables"
fi

echo ""
echo "🎉 DVC setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your AWS credentials: nano .env"
echo "2. Run the pipeline: dvc repro"
echo "3. Push data to S3: dvc push"
echo ""
echo "🔒 Security reminders:"
echo "- Never commit .env file to Git"
echo "- Use IAM roles in production"
echo "- Rotate credentials regularly"