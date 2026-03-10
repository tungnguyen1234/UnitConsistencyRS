import sys
import os
import json
from datetime import datetime


def save_data(json_file_path, data):
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)


def load_json(path: str):
    """
    Load a JSON configuration file with cross-platform path support.
    Expands '~' and environment variables, and normalizes separators.

    Args:
        path (str): Path to the JSON config file.
    Returns:
        dict: Parsed JSON data.
    """
    path = os.path.expanduser(os.path.expandvars(path))
    path = os.path.normpath(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


class Logger:
    def __init__(self, log_file_path):
        os.makedirs(os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else '.', exist_ok=True)
        self.log_file = open(log_file_path, 'a', encoding='utf-8', buffering=1)
        self.original_stdout = sys.stdout
        sys.stdout = self
        self.log_file_path = log_file_path

    def write(self, message):
        self.log_file.write(message)
        self.original_stdout.write(message)
        self.flush()

    def log(self, message):
        if not message.endswith('\n'):
            message += '\n'
        self.write(message)

    def flush(self):
        self.log_file.flush()
        self.original_stdout.flush()
        os.fsync(self.log_file.fileno())

    def close(self):
        sys.stdout = self.original_stdout
        self.log_file.close()

    def __getattr__(self, attr):
        return getattr(self.original_stdout, attr)
