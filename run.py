import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    os.chdir(pathlib.Path(__file__).parent)
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
