"""
Simple logging configuration for the LangChain RAG application.
"""

import logging
import os


def get_logger(name: str = None) -> logging.Logger:
    """Get a simple logger with your preferred format."""
    # Get log level from environment or default to INFO
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Create logger
    logger = logging.getLogger(name or "app")
    logger.setLevel(getattr(logging, level, logging.INFO))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Simple formatter: time | level | message (preserving emojis)
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


# Simple logger instance
logger = get_logger()