# scripts/run_app.py

import sys
import os

# Add the project root to the Python path
# This is crucial for imports like 'from src.embedding.vector_index import VectorIndex' to work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_root)

from src.gui.chatpdf_ui import ChatPDFApp
from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    ex = ChatPDFApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()