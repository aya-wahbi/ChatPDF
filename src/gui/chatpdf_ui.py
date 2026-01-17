import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QListWidget, QLabel, QFileDialog,
    QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Import your backend logic
from src.query.answer_generation import ChatPDFAssistant
from scripts.ingest_documents import main as ingest_documents_script # Import the ingestion script's main function
import os

# --- Worker Thread for LLM Calls (to keep UI responsive) ---
class QueryWorker(QThread):
    finished = pyqtSignal(dict) # Signal to emit the result (answer, sources)
    error = pyqtSignal(str)     # Signal to emit error messages

    def __init__(self, assistant: ChatPDFAssistant, query: str):
        super().__init__()
        self.assistant = assistant
        self.query = query

    def run(self):
        try:
            response = self.assistant.query_documents(self.query)
            self.finished.emit(response)
        except Exception as e:
            self.error.emit(f"Error during query: {e}")

# --- Main Application Window ---
class ChatPDFApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChatPDF - Generative AI Research Assistant")
        self.setGeometry(100, 100, 1000, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.layout.addWidget(self.chat_history)

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Ask a question about your documents...")
        self.query_input.returnPressed.connect(self.send_query) # Connect Enter key
        self.layout.addWidget(self.query_input)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_query)
        self.layout.addWidget(self.send_button)

        # --- Document Management Section ---
        self.doc_management_layout = QHBoxLayout()
        self.layout.addLayout(self.doc_management_layout)

        self.doc_list_label = QLabel("Loaded Documents:")
        self.doc_management_layout.addWidget(self.doc_list_label)

        self.doc_list = QListWidget()
        self.doc_list.setMaximumHeight(100)
        self.doc_management_layout.addWidget(self.doc_list)

        self.add_doc_button = QPushButton("Add Documents")
        self.add_doc_button.clicked.connect(self.add_documents)
        self.doc_management_layout.addWidget(self.add_doc_button)

        self.reindex_button = QPushButton("Re-index All")
        self.reindex_button.clicked.connect(self.reindex_documents)
        self.doc_management_layout.addWidget(self.reindex_button)
        
        # --- Backend Initialization ---
        self.index_dir = "data/index"
        self.index_file = os.path.join(self.index_dir, "faiss_index.bin")
        self.metadata_file = os.path.join(self.index_dir, "chunk_metadata.pkl")
        self.raw_data_dir = "data/raw"

        self.assistant = None
        self.load_assistant() # Try to load existing assistant on startup

        self.update_document_list() # Populate document list on startup

    def load_assistant(self):
        """Attempts to load the ChatPDFAssistant with existing index."""
        try:
            self.assistant = ChatPDFAssistant(self.index_file, self.metadata_file, llm_type='ollama', llm_model_name='llama2')
            self.chat_history.append("<font color='green'>Assistant loaded successfully!</font>")
        except FileNotFoundError:
            self.chat_history.append("<font color='orange'>No existing index found. Please add documents and re-index.</font>")
            self.assistant = None # Ensure assistant is None if loading fails
        except Exception as e:
            self.chat_history.append(f"<font color='red'>Error loading assistant: {e}</font>")
            self.assistant = None

    def update_document_list(self):
        """Refreshes the list of documents in the UI."""
        self.doc_list.clear()
        if os.path.exists(self.raw_data_dir):
            for file_name in os.listdir(self.raw_data_dir):
                if os.path.isfile(os.path.join(self.raw_data_dir, file_name)):
                    self.doc_list.addItem(file_name)
        if self.doc_list.count() == 0:
            self.doc_list.addItem("No documents added yet.")

    def add_documents(self):
        """Opens a file dialog to select and copy documents to raw data folder."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Documents", "", "PDF Files (*.pdf);;All Files (*)"
        )
        if file_paths:
            for src_path in file_paths:
                dest_path = os.path.join(self.raw_data_dir, os.path.basename(src_path))
                try:
                    import shutil
                    shutil.copy(src_path, dest_path)
                    self.chat_history.append(f"<font color='blue'>Added: {os.path.basename(src_path)}</font>")
                except Exception as e:
                    self.chat_history.append(f"<font color='red'>Error adding {os.path.basename(src_path)}: {e}</font>")
            self.update_document_list()
            QMessageBox.information(self, "Documents Added", "Documents copied to raw data folder. Click 'Re-index All' to process them.")

    def reindex_documents(self):
        """Triggers the ingestion and indexing process."""
        self.chat_history.append("<font color='purple'>Re-indexing documents... This may take a while.</font>")
        self.send_button.setEnabled(False)
        self.add_doc_button.setEnabled(False)
        self.reindex_button.setEnabled(False)
        
        # Run ingestion in a separate thread to keep UI responsive
        self.ingestion_worker = IngestionWorker()
        self.ingestion_worker.finished.connect(self.on_reindex_finished)
        self.ingestion_worker.error.connect(self.on_reindex_error)
        self.ingestion_worker.start()

    def on_reindex_finished(self):
        self.chat_history.append("<font color='green'>Document re-indexing complete!</font>")
        self.load_assistant() # Attempt to reload assistant with new index
        self.send_button.setEnabled(True)
        self.add_doc_button.setEnabled(True)
        self.reindex_button.setEnabled(True)

    def on_reindex_error(self, message):
        self.chat_history.append(f"<font color='red'>Re-indexing error: {message}</font>")
        self.send_button.setEnabled(True)
        self.add_doc_button.setEnabled(True)
        self.reindex_button.setEnabled(True)

    def send_query(self):
        query_text = self.query_input.text().strip()
        if not query_text:
            return

        self.chat_history.append(f"<p style='color: #007bff;'><b>You:</b> {query_text}</p>")
        self.query_input.clear()
        self.send_button.setEnabled(False) # Disable button while processing

        if not self.assistant:
            self.chat_history.append("<p style='color: red;'>Error: Assistant not loaded. Please re-index documents.</p>")
            self.send_button.setEnabled(True)
            return

        # Run query in a separate thread
        self.query_worker = QueryWorker(self.assistant, query_text)
        self.query_worker.finished.connect(self.display_answer)
        self.query_worker.error.connect(self.display_error)
        self.query_worker.start()

    def display_answer(self, response: Dict[str, Any]):
        answer = response['answer']
        sources = response['sources']
        
        self.chat_history.append(f"<p style='color: #333333;'><b>AI Assistant:</b> {answer}</p>")
        if sources:
            self.chat_history.append("<p style='color: #555555;'><b>Sources:</b></p>")
            for source in sources:
                self.chat_history.append(f"<p style='color: #555555; margin-left: 20px;'>- {source}</p>")
        
        self.send_button.setEnabled(True)

    def display_error(self, message: str):
        self.chat_history.append(f"<p style='color: red;'>Error: {message}</p>")
        self.send_button.setEnabled(True)

# --- Worker Thread for Ingestion (to keep UI responsive) ---
class IngestionWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def run(self):
        try:
            ingest_documents_script() # Call the main function of your ingestion script
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChatPDFApp()
    window.show()
    sys.exit(app.exec_())