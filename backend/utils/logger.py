# logger.py

import logging
import os
from datetime import datetime

class Logger:
    def __init__(self, name: str = "app_logger", log_dir: str = "logs", level: int = logging.INFO):
        """
        Initialize the logger.

        :param name: Name of the logger.
        :param log_dir: Directory where logs will be saved.
        :param level: Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # Prevent propagation to root logger

        # Avoid adding handlers multiple times
        if not self.logger.handlers:
            # Ensure the log directory exists
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y-%m-%d')}.log")

            # File handler ONLY (no console output)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)

            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s')
            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger

    def close(self):
        for handler in self.logger.handlers:
            handler.close()

    def info_multiline(self, message: str):
        """Log info message, preserving newlines."""
        self.logger.info(message)
    
    def error_multiline(self, message: str):
        """Log error message, preserving newlines."""
        self.logger.error(message)
    
    def debug_multiline(self, message: str):
        """Log debug message, preserving newlines."""
        self.logger.debug(message)
