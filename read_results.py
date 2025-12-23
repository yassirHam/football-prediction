import sys
import os

def read_file():
    filename = 'results_clean.txt'
    if not os.path.exists(filename):
        print("File not found.")
        return

    # Try utf-16 first (PowerShell default for >)
    try:
        with open(filename, 'r', encoding='utf-16') as f:
            print(f.read())
            return
    except UnicodeError:
        pass

    # Try utf-8
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            print(f.read())
            return
    except UnicodeError:
        pass
        
    # Fallback
    try:
        with open(filename, 'r', errors='ignore') as f:
            print(f.read())
    except:
        print("Could not read file.")

if __name__ == "__main__":
    read_file()
