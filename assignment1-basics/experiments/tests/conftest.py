import sys
from pathlib import Path

# experiments/ is not a package; put it on sys.path so `import trainer` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
