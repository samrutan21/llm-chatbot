import requests
import sys
import time
import subprocess
import platform

def check_ollama_running():
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
        else:
            print(f"❌ Ollama returned status code {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("❌ Ollama is not running or not accessible")
        return False

def check_llama_model():
    """Check if llama3.2 model is available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = [model["name"] for model in response.json().get("models", [])]
            if "llama3.2" in models:
                print("✅ Llama 3.2 model is available")
                return True
            else:
                print("❌ Llama 3.2 model is not available")
                return False
        return False
    except:
        print("❌ Cannot check models - Ollama not running")
        return False

def start_ollama():
    """Try to start Ollama if it's not running"""
    system = platform.system()
    try:
        if system == "Darwin" or system == "Linux":  # macOS or Linux
            subprocess.Popen(["ollama", "serve"], 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
            print("🔄 Attempting to start Ollama...")
            time.sleep(5)  # Give time for Ollama to start
            return check_ollama_running()
        elif system == "Windows":
            subprocess.Popen(["ollama", "serve"], 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE,
                            shell=True)
            print("🔄 Attempting to start Ollama...")
            time.sleep(5)  # Give time for Ollama to start
            return check_ollama_running()
        else:
            print(f"❌ Unsupported platform: {system}")
            return False
    except Exception as e:
        print(f"❌ Failed to start Ollama: {str(e)}")
        return False

def pull_llama_model():
    """Pull the llama3.2 model if it's not available"""
    try:
        print("🔄 Pulling Llama 3.2 model (this may take a while)...")
        subprocess.run(["ollama", "pull", "llama3.2"], check=True)
        return check_llama_model()
    except Exception as e:
        print(f"❌ Failed to pull Llama 3.2 model: {str(e)}")
        return False

def main():
    """Main function to check and set up Ollama with Llama 3.2"""
    print("Checking Ollama setup...")
    
    # Check if Ollama is running
    ollama_running = check_ollama_running()
    if not ollama_running:
        print("Ollama is not running. Attempting to start...")
        ollama_running = start_ollama()
        if not ollama_running:
            print("\n❌ Could not start Ollama. Please start it manually:")
            print("   - Make sure Ollama is installed: https://ollama.com/download")
            print("   - Run 'ollama serve' in a separate terminal")
            return False
    
    # Check if Llama 3.2 model is available
    llama_available = check_llama_model()
    if not llama_available:
        print("Llama 3.2 model is not available. Attempting to pull...")
        llama_available = pull_llama_model()
        if not llama_available:
            print("\n❌ Could not pull Llama 3.2 model. Please pull it manually:")
            print("   - Run 'ollama pull llama3.2' in a terminal")
            return False
    
    print("\n✅ Ollama is properly set up with Llama 3.2!")
    print("   You can now run your movie chatbot with:")
    print("   streamlit run bot.py")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)