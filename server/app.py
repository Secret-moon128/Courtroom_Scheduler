"""
server/app.py — openenv validate compatible entry point.
Imports the FastAPI app from the root server.py file.
"""
import sys
import os

# Make sure root is on path so we find server.py not server/
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

# Import app from root server.py explicitly by file path
import importlib.util
spec = importlib.util.spec_from_file_location("server_root", os.path.join(root, "server.py"))
server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server_module)
app = server_module.app

import uvicorn

def main():
    uvicorn.run("server:app", host="0.0.0.0", port=7860, app_dir=root)

if __name__ == "__main__":
    main()