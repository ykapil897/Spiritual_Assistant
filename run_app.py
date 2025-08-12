import subprocess
import time
import os
import signal
import sys

# Detect base path (works for exe and dev)
if getattr(sys, 'frozen', False):
    BASE_PATH = sys._MEIPASS
    print("Running in frozen mode, base path:", BASE_PATH)

    # Prepend bundled bin folder to PATH so uvicorn/streamlit can be found
    bin_path = os.path.join(BASE_PATH, "bin")
    os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
    print("Updated PATH:", os.environ["PATH"])
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    print("Running in development mode, base path:", BASE_PATH)

# Script names (will be inside _internal in exe mode)
BACKEND_SCRIPT = "api.py"
FRONTEND_SCRIPT = "main_streamlit_app.py"

if getattr(sys, 'frozen', False):
    os.chdir(BASE_PATH)

def kill_process_tree(pid):
    """Kill a process and all its children."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        pass

# Backend command (binary form)
backend_cmd = [
    "uvicorn",
    "api:app",
    "--port", "8000"
]
if not getattr(sys, 'frozen', False):
    backend_cmd.append("--reload")  # Only in dev mode

print(f"📜 Backend command: {' '.join(backend_cmd)}")
try:
    backend_proc = subprocess.Popen(
        backend_cmd,
        preexec_fn=os.setsid
    )
    print("✅ Backend started successfully (PID:", backend_proc.pid, ")")
except Exception as e:
    print("❌ Failed to start backend:", e)

time.sleep(3)  # Give backend time to start

# Frontend command
frontend_cmd = [
    "streamlit",
    "run",
    FRONTEND_SCRIPT
]

print(f"🎨 Frontend command: {' '.join(frontend_cmd)}")
try:
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        preexec_fn=os.setsid
    )
    print("✅ Frontend started successfully (PID:", frontend_proc.pid, ")")
except Exception as e:
    print("❌ Failed to start frontend:", e)

try:
    backend_proc.wait()
    frontend_proc.wait()
except KeyboardInterrupt:
    print("\n🛑 Shutting down...")
    kill_process_tree(backend_proc.pid)
    kill_process_tree(frontend_proc.pid)
