import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    python_version = sys.version_info
    
    if python_version.major == 3 and python_version.minor >= 12:
        print("Python 3.12+ detected. Using compatibility mode...")
        return "3.12+"
    elif python_version.major == 3 and python_version.minor >= 9:
        print(f"Python {python_version.major}.{python_version.minor} detected.")
        return "3.9+"
    else:
        print(f"Warning: Python {python_version.major}.{python_version.minor} might not be fully compatible.")
        print("Recommended: Python 3.9 or higher")
        return "legacy"

def create_directories():
    """Create necessary directories"""
    dirs = ["templates", "static", "uploads", "chroma_db"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created directory: {dir_name}")

def setup_environment():
    """Set up virtual environment and install dependencies"""
    python_version = check_python_version()
    
    # Create directories
    create_directories()
    
    # Install dependencies
    print("\nInstalling dependencies...")
    
    # Use a different approach for Python 3.12
    if python_version == "3.12+":
        # Install packages one by one to better handle compatibility issues
        packages = [
            "fastapi>=0.109.0",
            "uvicorn>=0.27.0",
            "python-multipart>=0.0.7",
            "pymupdf>=1.23.19",
            "python-dotenv>=1.0.0",
            "google-generativeai>=0.3.2",
            "--extra-index-url https://download.pytorch.org/whl/cpu torch>=2.1.0",
            "--extra-index-url https://download.pytorch.org/whl/cpu torchvision>=0.16.0",
        ]
        
        for package in packages:
            print(f"Installing {package}...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
            except subprocess.CalledProcessError:
                print(f"Warning: Failed to install {package}, trying without version constraint...")
                try:
                    # Try without version constraint
                    pkg_name = package.split(">=")[0]
                    subprocess.run([sys.executable, "-m", "pip", "install", pkg_name], check=True)
                except:
                    print(f"Error installing {pkg_name}. Please install it manually.")

        # Install sentence-transformers and chromadb separately as they can be problematic
        print("Installing sentence-transformers...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "sentence-transformers>=2.2.2"], check=True)
        except:
            print("Warning: Issue installing sentence-transformers. Trying without version constraint...")
            subprocess.run([sys.executable, "-m", "pip", "install", "sentence-transformers"], check=True)
            
        print("Installing chromadb...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "chromadb>=0.4.22"], check=True)
        except:
            print("Warning: Issue installing chromadb. Trying without version constraint...")
            subprocess.run([sys.executable, "-m", "pip", "install", "chromadb"], check=True)
    else:
        # For Python 3.9-3.11, we can use the normal pip install
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        except subprocess.CalledProcessError:
            print("Error installing dependencies. Please check the error messages above.")
            sys.exit(1)
    
    print("\nSetup completed successfully!")
    print("Now update your config.env file with your Gemini API key.")
    print("Then run 'python app.py' to start the application.")

if __name__ == "__main__":
    setup_environment()