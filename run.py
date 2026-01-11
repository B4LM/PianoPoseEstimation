import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.main import main
    
    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"Error: {e}")
    print("\nMake sure to install dependencies first:")
    print("pip install -r requirements.txt")
    sys.exit(1)
