"""
Entrypoint for Streamlit Community Cloud: runs the project's app.py.

It exists only so Community Cloud picks up streamlit_cloud/requirements.txt
(CPU TensorFlow) instead of the root one. The hosted demo ships just
MobileNetV2 and no database: contributions and sign-ins are switched off
because no TURSO_* secrets are set.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
