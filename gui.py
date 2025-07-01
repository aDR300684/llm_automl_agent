# gui.py

import os
import shutil
import sys
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QComboBox, QTextEdit, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal, QObject


class AutoMLWorker(QObject):
    finished = Signal(str)

    def __init__(self, csv_path, target_col):
        super().__init__()
        self.csv_path = csv_path
        self.target_col = target_col

    def run(self):
        from llm_agent import run_llm_pipeline  # Delayed import for performance
        result = run_llm_pipeline(self.csv_path, self.target_col)
        self.finished.emit(result)


class AutoMLGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 AutoML Agent (LLM-powered)")
        self.resize(600, 500)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.upload_button = QPushButton("📂 Load CSV file")
        self.upload_button.clicked.connect(self.load_csv)
        self.layout.addWidget(self.upload_button)

        self.label = QLabel("🎯 Target column:")
        self.layout.addWidget(self.label)

        self.target_selector = QComboBox()
        self.layout.addWidget(self.target_selector)

        self.run_button = QPushButton("🚀 Run AutoML Agent")
        self.run_button.clicked.connect(self.run_agent)
        self.layout.addWidget(self.run_button)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setPlaceholderText("📜 Agent logs will appear here...")
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #f0f0f0;
                font-family: Segoe UI, sans-serif;
                font-size: 13px;
            }
            QPushButton {
                background-color: #3c3f41;
                color: white;
                border: 1px solid #555;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #505354;
            }
            QComboBox {
                background-color: #3c3f41;
                color: white;
                border: 1px solid #555;
                padding: 4px;
            }
            QLabel {
                color: #ffcc66;
                font-weight: bold;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #dcdcdc;
                font-family: Consolas, monospace;
                font-size: 12px;
            }
        """)
        self.layout.addWidget(self.console)

        self.csv_path = None
        self.df = None
        self.thread = None
        self.worker = None

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a CSV file", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.csv_path = file_path
                self.target_selector.clear()
                self.target_selector.addItems(self.df.columns.tolist())

                # 🔁 Copy to /data/
                dest_path = os.path.join("data", os.path.basename(file_path))
                shutil.copy(file_path, dest_path)
                self.csv_path = dest_path  # update path
                self.log(f"[INFO] File copied to /data: {dest_path}")
                self.log(f"[INFO] Detected {self.df.shape[0]} rows, {self.df.shape[1]} columns.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSV: {str(e)}")

    def run_agent(self):
        if not self.csv_path or self.df is None:
            QMessageBox.warning(self, "Warning", "Please load a CSV file first.")
            return

        target = self.target_selector.currentText()
        if not target:
            QMessageBox.warning(self, "Warning", "Please select a target column.")
            return

        self.log(f"\n[RUN] Running AutoML agent on target column: '{target}'...")
        self.log("[INFO] Launching background LLM agent. This may take up to a minute if GPT is slow...")


        # Start worker in new thread
        self.thread = QThread()
        self.worker = AutoMLWorker(self.csv_path, target)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.handle_pipeline_result)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def handle_pipeline_result(self, result: str):
        self.log(result)

    def log(self, message):
        self.console.append(message)
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AutoMLGUI()
    window.show()
    sys.exit(app.exec())
