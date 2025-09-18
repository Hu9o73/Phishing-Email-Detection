import subprocess
import sys

def clear_console():
    subprocess.run('cls' if sys.platform == 'win32' else 'clear', shell=True)
