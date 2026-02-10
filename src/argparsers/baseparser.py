import argparse

from utils.logging import get_logger

logger = get_logger(__name__)

class BaseParser:
    def __init__(self, description: str):
        self.parser = argparse.ArgumentParser(description=description)
        self.parser.add_argument("--log-level", "-L", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                                 default="INFO", help="Set the logging level")
        
    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)
        
    def parse_args(self, args=None):
        return self.parser.parse_args(args)
