#!/usr/bin/env bash

# Copy this file to scripts/export_env.sh and fill in your actual values.
# Keep scripts/export_env.sh out of version control.

# Azure OpenAI Configuration
export AZURE_OPENAI_API_KEY="your_azure_openai_api_key_here"
export AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"

# External API Keys for Tools
export GOOGLE_MAP_API_KEY="your_google_maps_api_key_here"
export AMAP_API_KEY="your_amap_api_key_here"
export LOCATIONIQ_API_KEY="your_locationiq_api_key_here"
export SERPAPI_KEY="your_serpapi_key_here"
export GOOGLE_CALENDAR_ACCOUNT="your_google_calendar_account_here"

# Optional: GPU configuration
# export CUDA_VISIBLE_DEVICES=0,1

# Usage:
#   source scripts/export_env.sh


