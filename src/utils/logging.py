import logging
from typing import Optional
from colorama import Fore, Style, init

# Initialize colorama
init()
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*glibc.*older than 2\.28.*",
    category=FutureWarning,
    module=r"xgboost\.core",
)



class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages"""
    
    COLORS = {
        'DEBUG': Fore.BLUE,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)

def setup_logging(log_level: str = 'INFO') -> None:
    """Configure logging with the specified level and colors"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create console handler with colored formatter
    console_handler = logging.StreamHandler()
    colored_formatter = ColoredFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(colored_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('art').setLevel(logging.ERROR)
    logging.getLogger('numba').setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger for the specified module"""
    return logging.getLogger(name) 