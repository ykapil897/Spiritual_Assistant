import subprocess
import os
import sys

# This ensures compatibility in PyInstaller
script_path = os.path.join(os.path.dirname(__file__), 'main_streamlit_app.py')

subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])
