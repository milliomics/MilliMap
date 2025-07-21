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
    # Set application icon globally using relative path
    # Get the directory containing this file (millimap/Code/)
    current_dir = pathlib.Path(__file__).parent
    # Navigate to millimap/Icons/cakeinvert.png
    icon_path = current_dir.parent / "Icons" / "cakeinvert.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    viewer = MillimapViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main() 