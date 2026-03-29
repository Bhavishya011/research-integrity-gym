"""Helper script to create OpenEnv required files."""
import os

base = r"c:\Users\rockb\OneDrive\Desktop\Projects\Meta Hack\research-integrity-gym"

# Create server directory
server_dir = os.path.join(base, "server")
os.makedirs(server_dir, exist_ok=True)
print(f"Created directory: {server_dir}")

# Create server/__init__.py
init_file = os.path.join(server_dir, "__init__.py")
with open(init_file, "w") as f:
    f.write('"""OpenEnv server module."""\n')
print(f"Created: {init_file}")

# Create server/app.py
app_content = '''"""
OpenEnv server entry point.

This module provides the main() function for the 'server' script entry point
required by OpenEnv multi-mode deployment.
"""
import uvicorn


def main():
    """Run the FastAPI server."""
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )


if __name__ == "__main__":
    main()
'''
app_file = os.path.join(server_dir, "app.py")
with open(app_file, "w") as f:
    f.write(app_content)
print(f"Created: {app_file}")

print("\nDone! Files created successfully.")
print("\nNext steps:")
print("1. pip install uv")
print("2. uv lock")
print("3. git add -A")
print("4. git commit -m 'Add OpenEnv multi-mode deployment requirements'")
print("5. git push hf main")
