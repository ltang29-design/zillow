#!/usr/bin/env python3
"""
Quick Setup Script for Housing Market Prediction System
Helps users configure the environment and get started quickly.
"""

import os
import sys
from pathlib import Path
import secrets

def create_env_file():
    """Create .env file from template."""
    env_template = Path("env.template")
    env_file = Path(".env")
    
    if env_file.exists():
        print("📝 .env file already exists")
        return
    
    if not env_template.exists():
        print("❌ env.template not found")
        return
    
    # Read template
    with open(env_template, 'r') as f:
        content = f.read()
    
    # Generate a secure secret key
    secret_key = secrets.token_urlsafe(32)
    content = content.replace('your-super-secret-key-change-this-in-production', secret_key)
    
    # Get FRED API key from user
    print("\n🔑 FRED API Key Setup:")
    print("You need a free FRED API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
    fred_key = input("Enter your FRED API key (or press Enter to skip): ").strip()
    
    if fred_key:
        content = content.replace('your_fred_api_key_here', fred_key)
        print("✅ FRED API key configured")
    else:
        print("⚠️  FRED API key not set - you can add it later to .env file")
    
    # Write .env file
    with open(env_file, 'w') as f:
        f.write(content)
    
    print("✅ Created .env file")

def check_data_directory():
    """Check if data directory exists and has CSV files."""
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("📁 Creating data directory...")
        data_dir.mkdir()
        print("✅ Data directory created")
        print("ℹ️  Place your Zillow CSV files in the 'data' directory")
        return False
    
    csv_files = list(data_dir.glob("Metro_*.csv"))
    if csv_files:
        print(f"✅ Found {len(csv_files)} Zillow CSV files in data directory")
        return True
    else:
        print("⚠️  No Zillow CSV files found in data directory")
        print("ℹ️  Place your Metro_*.csv files in the 'data' directory")
        return False

def create_directories():
    """Create necessary directories."""
    dirs = ["logs", "models", "data/features"]
    
    for dir_path in dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True)
            print(f"✅ Created directory: {dir_path}")

def install_dependencies():
    """Check if dependencies are installed."""
    try:
        import fastapi
        import sqlalchemy
        import pandas
        print("✅ Core dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False

def main():
    """Main setup function."""
    print("🏠 Housing Market Prediction System - Quick Setup")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("cli.py").exists():
        print("❌ Error: cli.py not found. Please run from project root directory.")
        sys.exit(1)
    
    # Install dependencies check
    print("\n1️⃣ Checking dependencies...")
    deps_ok = install_dependencies()
    
    # Create directories
    print("\n2️⃣ Creating directories...")
    create_directories()
    
    # Create .env file
    print("\n3️⃣ Setting up environment...")
    create_env_file()
    
    # Check data directory
    print("\n4️⃣ Checking data directory...")
    has_data = check_data_directory()
    
    # Summary and next steps
    print("\n" + "=" * 60)
    print("📋 SETUP SUMMARY")
    print("=" * 60)
    
    if deps_ok:
        print("✅ Dependencies: Ready")
    else:
        print("❌ Dependencies: Need to install")
        
    print("✅ Directories: Created")
    print("✅ Environment: Configured")
    
    if has_data:
        print("✅ Data: CSV files found")
    else:
        print("⚠️  Data: No CSV files (optional for basic testing)")
    
    print("\n🚀 NEXT STEPS:")
    print("1. If dependencies failed: pip install -r requirements.txt")
    print("2. Put your FRED API key in .env file (FRED_API_KEY=your_key)")
    print("3. Place Zillow CSV files in data/ directory (optional)")
    print("4. Test the system: python test_system.py")
    print("5. Or run manual tests: python cli.py health-check")
    
    print("\n📖 QUICK START COMMANDS:")
    print("  python cli.py config-info        # Check configuration")
    print("  python cli.py init-db            # Initialize database")
    print("  python cli.py health-check       # Test system health")
    print("  python cli.py ingest-data        # Load Zillow data")
    print("  python cli.py start-api          # Start web API")
    
    print("\n🎉 Setup complete! Ready to start testing.")

if __name__ == "__main__":
    main() 