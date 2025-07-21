import sys
import pathlib
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

# Handle both direct execution and package import
try:
    # Try relative import first (when used as package)
    from .viewer import MillimapViewer
except (ImportError, ValueError):
    # Fall back to absolute import (when run directly or imported directly)
    try:
        from viewer import MillimapViewer
    except ImportError:
        # If that fails, try adding current directory to path
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from viewer import MillimapViewer


def main() -> None:
    app = QApplication(sys.argv)
    # Set application icon globally
    icon_path = "millimap/Icons/cakeinvert.png"
    if pathlib.Path(icon_path).exists():
        app.setWindowIcon(QIcon(icon_path))
    viewer = MillimapViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 