#!/bin/bash

echo "======================================"
echo "  Chenex v1.1.3 Startup Script"
echo "  Powered by Relaquent"
echo "======================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

echo "✓ Python3 found"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Start the application
echo ""
echo "🚀 Starting Chenex v1.1.3..."
echo ""
python3 app.py