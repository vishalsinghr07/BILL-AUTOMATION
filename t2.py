import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QLabel,
    QLineEdit, QProgressBar, QMessageBox, QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import QThread, QObject, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QIcon

# --- Third-party libraries ---
# You must install these first by running the following command in your terminal:
# pip install PyQt5 pandas pdfplumber google-generativeai

import pandas as pd
import pdfplumber
import google.generativeai as genai

# =============================================================================
# CONFIGURATION
# IMPORTANT: Replace "YOUR_API_KEY_HERE" with your actual Gemini API key.
# This key will be embedded in the application.
# =============================================================================
GEMINI_API_KEY = "AIzaSyAkDmTuK7ZT1ZWwoKyJT4WQZpF6OZ9g5h4"

# =============================================================================
# WORKER THREAD FOR BACKGROUND PROCESSING
# =============================================================================
class ExtractorWorker(QObject):
    """
    Worker thread to perform PDF extraction without freezing the GUI.
    """
    progress = pyqtSignal(int)
    row_extracted = pyqtSignal(dict)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, api_key, pdf_paths):
        super().__init__()
        self.api_key = api_key
        self.pdf_paths = pdf_paths
        self.is_running = True

    def run(self):
        """Main processing loop."""
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
        except Exception as e:
            self.error.emit(f"Failed to configure Gemini API. Check your API key.\nError: {e}")
            return

        total_files = len(self.pdf_paths)
        for i, pdf_path in enumerate(self.pdf_paths):
            if not self.is_running:
                break
            
            # 1. Read PDF content
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    full_text = "".join(page.extract_text() for page in pdf.pages)
                if not full_text.strip():
                    self.error.emit(f"Could not extract text from: {os.path.basename(pdf_path)}")
                    self.progress.emit(int(((i + 1) / total_files) * 100))
                    continue
            except Exception as e:
                self.error.emit(f"Error reading {os.path.basename(pdf_path)}.\nError: {e}")
                self.progress.emit(int(((i + 1) / total_files) * 100))
                continue

            # 2. Create prompt and call Gemini API
            prompt = self.create_prompt(full_text)
            try:
                response = model.generate_content(prompt)
                json_text = response.text.strip().replace('```json', '').replace('```', '')
                extracted_data = json.loads(json_text)
                extracted_data['FileName'] = os.path.basename(pdf_path)
                self.row_extracted.emit(extracted_data)
            except Exception as e:
                self.error.emit(f"Failed to process {os.path.basename(pdf_path)} with Gemini.\nError: {e}")
            
            # 3. Update progress
            self.progress.emit(int(((i + 1) / total_files) * 100))

        self.finished.emit()

    def create_prompt(self, text):
        """Creates the detailed prompt for the Gemini API."""
        return f"""
        Analyze the following invoice text and extract the specified fields.
        Respond ONLY with a single, clean JSON object. Do not add any explanatory text before or after the JSON.

        The fields to extract are:
        - "VendorName": The name of the company that sent the invoice.
        - "InvoiceNumber": The unique identifier for the invoice.
        - "InvoiceDate": The date the invoice was issued.
        - "Product": A brief description of the product or service. If multiple, list the main one.
        - "SerialNumber": The serial or product number, if available.
        - "GST": The total Goods and Services Tax amount.
        - "CGST": The Central GST amount, if specified separately.
        - "Total": The final, total amount due on the invoice.

        If a value is not found for any field, use "N/A" as the value.

        Here is the invoice text:
        ---
        {text}
        ---
        """
        
    def stop(self):
        self.is_running = False

# =============================================================================
# MAIN GUI APPLICATION WINDOW
# =============================================================================
class InvoiceExtractorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Invoice Data Extractor (powered by Gemini)")
        self.setGeometry(100, 100, 1200, 800)
        self.pdf_files = []
        self.column_headers = [
            "FileName", "VendorName", "InvoiceNumber", "InvoiceDate", "Product", 
            "SerialNumber", "GST", "CGST", "Total"
        ]
        self.init_ui()

    def init_ui(self):
        """Sets up the entire user interface."""
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        title_label = QLabel("Invoice Data Extractor")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # --- API Key Input has been removed from the UI ---

        button_layout = QHBoxLayout()
        self.select_button = QPushButton("1. Select Invoice PDFs")
        self.select_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.select_button.clicked.connect(self.select_files)
        
        self.start_button = QPushButton("2. Start Extraction")
        self.start_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_extraction)
        
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.start_button)
        main_layout.addLayout(button_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        self.files_label = QLabel("No files selected.")
        self.files_label.setFont(QFont("Arial", 10, italic=True))
        main_layout.addWidget(self.files_label)

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(len(self.column_headers))
        self.table_widget.setHorizontalHeaderLabels(self.column_headers)
        self.table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        main_layout.addWidget(self.table_widget)

        self.export_button = QPushButton("Export to CSV")
        self.export_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_to_csv)
        main_layout.addWidget(self.export_button, alignment=Qt.AlignRight)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QPushButton { 
                background-color: #007BFF; color: white; padding: 10px; 
                border-radius: 5px; border: none;
            }
            QPushButton:hover { background-color: #0056b3; }
            QPushButton:disabled { background-color: #a0a0a0; }
            QTableWidget { border: 1px solid #ccc; gridline-color: #d0d0d0; }
            QHeaderView::section { background-color: #e0e0e0; padding: 5px; border: 1px solid #ccc; }
            QLabel { color: #333; }
        """)

    def select_files(self):
        """Opens a dialog to select multiple PDF files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Invoice PDF Files", "", "PDF Files (*.pdf)"
        )
        if files:
            self.pdf_files = files
            self.files_label.setText(f"{len(self.pdf_files)} file(s) selected.")
            self.start_button.setEnabled(True)
            self.table_widget.setRowCount(0)
            self.export_button.setEnabled(False)

    def start_extraction(self):
        """Validates input and starts the background extraction process."""
        # --- The API key is now taken from the constant defined at the top ---
        if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_API_KEY_HERE":
            QMessageBox.critical(self, "API Key Missing", 
                                 "The Gemini API key has not been set in the code.\n"
                                 "Please ask the developer to add it to the script.")
            return

        if not self.pdf_files:
            QMessageBox.warning(self, "No Files", "Please select PDF files to process.")
            return

        self.start_button.setEnabled(False)
        self.select_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.table_widget.setRowCount(0)

        self.thread = QThread()
        self.worker = ExtractorWorker(GEMINI_API_KEY, self.pdf_files)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_extraction_finished)
        self.worker.progress.connect(self.set_progress)
        self.worker.row_extracted.connect(self.add_row_to_table)
        self.worker.error.connect(self.show_error_message)

        self.thread.start()

    def add_row_to_table(self, data):
        """Adds a new row of extracted data to the results table."""
        row_position = self.table_widget.rowCount()
        self.table_widget.insertRow(row_position)
        for col_index, col_name in enumerate(self.column_headers):
            item_value = str(data.get(col_name, 'N/A'))
            self.table_widget.setItem(row_position, col_index, QTableWidgetItem(item_value))
    
    def set_progress(self, value):
        self.progress_bar.setValue(value)

    def on_extraction_finished(self):
        self.thread.quit()
        self.thread.wait()
        self.progress_bar.setVisible(False)
        self.start_button.setEnabled(True)
        self.select_button.setEnabled(True)
        if self.table_widget.rowCount() > 0:
            self.export_button.setEnabled(True)
        QMessageBox.information(self, "Success", "Data extraction complete!")

    def show_error_message(self, message):
        QMessageBox.warning(self, "Processing Error", message)

    def export_to_csv(self):
        if self.table_widget.rowCount() == 0:
            QMessageBox.warning(self, "No Data", "There is no data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv)")
        if path:
            try:
                data = []
                for row in range(self.table_widget.rowCount()):
                    row_data = {self.table_widget.horizontalHeaderItem(col).text(): 
                                self.table_widget.item(row, col).text() if self.table_widget.item(row, col) else "" 
                                for col in range(self.table_widget.columnCount())}
                    data.append(row_data)
                
                df = pd.DataFrame(data)
                df.to_csv(path, index=False)
                QMessageBox.information(self, "Success", f"Data successfully exported to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export data.\nError: {e}")

    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker:
            self.worker.stop()
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        event.accept()

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = InvoiceExtractorApp()
    main_app.show()
    sys.exit(app.exec_())
