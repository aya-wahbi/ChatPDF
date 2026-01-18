# src/gui/chatpdf_ui.py

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel,
    QMessageBox, QScrollArea, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor

# Import your backend logic
from src.embedding.vector_index import VectorIndex
from src.query.answer_generation import ChatPDFAssistant

# --- Worker Thread for LLM Querying ---
# This is crucial for keeping the UI responsive while the LLM processes the query.
class QueryWorker(QThread):
    # Signals to communicate with the main thread
    query_started = pyqtSignal(str)
    query_finished = pyqtSignal(dict)
    query_error = pyqtSignal(str)

    def __init__(self, assistant: ChatPDFAssistant, query_text: str):
        super().__init__()
        self.assistant = assistant
        self.query_text = query_text

    def run(self):
        try:
            self.query_started.emit(self.query_text)
            response = self.assistant.query_documents(self.query_text)
            self.query_finished.emit(response)
        except Exception as e:
            self.query_error.emit(f"An error occurred during query: {e}")

# --- Main GUI Application ---
class ChatPDFApp(QWidget):
    def __init__(self):
        super().__init__()
        self.assistant = None  # Will be initialized after loading index
        self.init_ui()
        self.load_index_and_assistant()

    def init_ui(self):
        self.setWindowTitle("ChatPDF - Research Assistant")
        self.setGeometry(100, 100, 900, 700) # x, y, width, height

        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # --- Header ---
        header_label = QLabel("Chat with Your Research Papers")
        header_label.setFont(QFont("Arial", 20, QFont.Bold))
        header_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header_label)

        # --- Status Bar ---
        self.status_label = QLabel("Initializing...")
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setStyleSheet("color: gray;")
        main_layout.addWidget(self.status_label)

        # --- Chat History / Output Area ---
        self.chat_history_text = QTextEdit()
        self.chat_history_text.setReadOnly(True)
        self.chat_history_text.setFont(QFont("Arial", 11))
        self.chat_history_text.setPlaceholderText("Your conversation with the AI will appear here...")
        self.chat_history_text.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; padding: 10px;")
        
        # Make the chat history scrollable
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.chat_history_text)
        main_layout.addWidget(scroll_area, 1) # Stretch factor 1

        # --- Input Area ---
        input_layout = QHBoxLayout()
        
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Ask a question about your documents...")
        self.query_input.setFont(QFont("Arial", 11))
        self.query_input.returnPressed.connect(self.send_query) # Connect Enter key to send_query
        input_layout.addWidget(self.query_input)

        self.send_button = QPushButton("Send")
        self.send_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.send_button.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 5px; padding: 8px;")
        self.send_button.clicked.connect(self.send_query)
        input_layout.addWidget(self.send_button)

        main_layout.addLayout(input_layout)

        # --- Separator ---
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # --- Footer ---
        footer_label = QLabel("Powered by Ollama and Sentence Transformers")
        footer_label.setFont(QFont("Arial", 9))
        footer_label.setAlignment(Qt.AlignRight)
        footer_label.setStyleSheet("color: #777;")
        main_layout.addWidget(footer_label)

        self.set_ui_enabled(False) # Disable UI until assistant is loaded

    def set_ui_enabled(self, enabled: bool):
        self.query_input.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
        if enabled:
            self.query_input.setPlaceholderText("Ask a question about your documents...")
        else:
            self.query_input.setPlaceholderText("Loading assistant... Please wait.")

    def load_index_and_assistant(self):
        index_dir = "data/index"
        index_file = os.path.join(index_dir, "faiss_index.bin")
        metadata_file = os.path.join(index_dir, "chunk_metadata.pkl")

        self.status_label.setText("Loading vector index and LLM assistant...")
        QApplication.processEvents() # Update UI immediately

        try:
            self.assistant = ChatPDFAssistant(index_file, metadata_file, llm_type='ollama', llm_model_name='llama2')
            self.status_label.setText("Assistant ready! You can now ask questions.")
            self.set_ui_enabled(True)
        except FileNotFoundError:
            self.status_label.setText("Error: Index files not found. Please run 'scripts/ingest_documents.py' first.")
            QMessageBox.critical(self, "Error", "Vector index files not found. Please run 'scripts/ingest_documents.py' to ingest your documents before launching the GUI.")
        except Exception as e:
            self.status_label.setText(f"Error initializing assistant: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred while initializing the assistant: {e}\nPlease ensure Ollama is running and 'llama2' model is downloaded.")
        
    def send_query(self):
        query_text = self.query_input.text().strip()
        if not query_text:
            return

        self.query_input.clear()
        self.set_ui_enabled(False) # Disable input while processing

        # Append user query to chat history
        self.append_to_chat_history(f"<b>You:</b> {query_text}\n", user_query=True)
        self.status_label.setText("Querying LLM... This may take a moment.")

        # Start worker thread
        self.worker = QueryWorker(self.assistant, query_text)
        self.worker.query_started.connect(self.on_query_started)
        self.worker.query_finished.connect(self.on_query_finished)
        self.worker.query_error.connect(self.on_query_error)
        self.worker.start()

    def on_query_started(self, query_text):
        # This signal is emitted from the worker, but we already added the user query
        pass 

    def on_query_finished(self, response: dict):
        answer = response.get('answer', 'No answer received.')
        sources = response.get('sources', [])

        self.append_to_chat_history(f"<b>AI Assistant:</b> {answer}\n", ai_response=True)
        if sources:
            self.append_to_chat_history("<b>Sources:</b>\n", is_source_header=True)
            for source in sources:
                self.append_to_chat_history(f"  - {source}\n", is_source=True)
        
        self.status_label.setText("Assistant ready! You can now ask questions.")
        self.set_ui_enabled(True)

    def on_query_error(self, error_message: str):
        self.append_to_chat_history(f"<b style='color: red;'>Error:</b> {error_message}\n", is_error=True)
        self.status_label.setText(f"Error: {error_message}")
        QMessageBox.critical(self, "Query Error", error_message)
        self.set_ui_enabled(True)

    def append_to_chat_history(self, text: str, user_query=False, ai_response=False, is_source_header=False, is_source=False, is_error=False):
        # Move cursor to end before appending
        cursor = self.chat_history_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_history_text.setTextCursor(cursor)

        if user_query:
            self.chat_history_text.append(f"<p style='color: #333; margin-bottom: 5px;'>{text}</p>")
        elif ai_response:
            self.chat_history_text.append(f"<p style='color: #0056b3; margin-top: 5px; margin-bottom: 10px;'>{text}</p>")
        elif is_source_header:
            self.chat_history_text.append(f"<p style='font-weight: bold; color: #555; margin-top: 10px; margin-bottom: 3px;'>{text}</p>")
        elif is_source:
            self.chat_history_text.append(f"<p style='font-size: 0.9em; color: #777; margin-left: 20px; margin-bottom: 2px;'>{text}</p>")
        elif is_error:
            self.chat_history_text.append(f"<p style='color: red; font-weight: bold;'>{text}</p>")
        else:
            self.chat_history_text.append(text)
        
        # Scroll to the bottom
        self.chat_history_text.verticalScrollBar().setValue(self.chat_history_text.verticalScrollBar().maximum())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ChatPDFApp()
    ex.show()
    sys.exit(app.exec_())