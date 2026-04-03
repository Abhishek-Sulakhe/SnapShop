import uvicorn
import os
import sys

if __name__ == "__main__":
    # Ensure we are in the webapp directory context
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Starting PriceMatch AI Web Server...")
    print("Open your browser at http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)