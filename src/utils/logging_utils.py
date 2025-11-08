"""
Logging utilities
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str,
    log_file: str = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        log_file: Optional file path for logging
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Format
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    """Track experiment progress"""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        
        self.logger = setup_logger(
            name=experiment_name,
            log_file=str(log_file),
            level=logging.INFO
        )
        
        self.logger.info(f"Experiment: {experiment_name}")
        self.logger.info(f"Timestamp: {datetime.now()}")
    
    def log_config(self, config: dict):
        """Log configuration"""
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_epoch(self, epoch: int, metrics: dict):
        """Log epoch metrics"""
        msg = f"Epoch {epoch}: "
        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f"{key}={value:.4f}, "
            else:
                msg += f"{key}={value}, "
        self.logger.info(msg)
    
    def log_result(self, metric_name: str, value: float):
        """Log single metric"""
        self.logger.info(f"{metric_name}: {value:.4f}")
    
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(message)
