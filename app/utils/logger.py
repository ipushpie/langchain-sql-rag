"""
Logging configuration for the LangChain RAG application.
Provides a centralized logging setup with timestamps and proper formatting.
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = __name__,
    level: str = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with proper formatting and timestamps.
    
    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to log to file as well as console
        
    Returns:
        Configured logger instance
    """
    # Get log level from environment or default to INFO
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level, logging.INFO))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatter with timestamp
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level, logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level, logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create a default logger for the application
app_logger = setup_logger("langchain_ragflow")


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance with proper configuration.
    
    Args:
        name: Logger name, defaults to calling module
        
    Returns:
        Configured logger instance
    """
    if name is None:
        # Get the calling module name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'unknown')
        else:
            name = 'unknown'
    
    return setup_logger(name)


# Emoji to log level mapping for better visual consistency
EMOJI_LOG_MAPPING = {
    '🧠': 'INFO',     # Model/AI operations
    '🔍': 'DEBUG',    # Analysis/search operations  
    '✅': 'INFO',     # Success operations
    '⚠️': 'WARNING',  # Warning conditions
    '❌': 'ERROR',    # Error conditions
    '🔄': 'INFO',     # Retry/fallback operations
    '🤖': 'DEBUG',    # LLM responses
    '🎯': 'DEBUG',    # Selection operations
    '📋': 'DEBUG',    # Formatting operations
    '🧹': 'DEBUG',    # Cleaning operations
    '🔧': 'DEBUG',    # Processing operations
    '🗄️': 'INFO',     # Database operations
    '🧮': 'INFO',     # SQL generation
    '🧭': 'DEBUG',    # Navigation/routes
    '📊': 'INFO',     # Results/statistics
    '👋': 'INFO',     # Greeting/interaction
    '👤': 'INFO',     # User/customer operations
}


def log_with_emoji(logger: logging.Logger, message: str, default_level: str = 'INFO'):
    """
    Log a message while detecting emoji prefix to determine appropriate log level.
    
    Args:
        logger: Logger instance
        message: Message to log (may start with emoji)
        default_level: Default log level if no emoji detected
    """
    # Extract emoji from start of message
    emoji = None
    clean_message = message
    
    for emoji_char in EMOJI_LOG_MAPPING:
        if message.startswith(emoji_char):
            emoji = emoji_char
            clean_message = message[len(emoji_char):].strip()
            break
    
    # Determine log level
    if emoji and emoji in EMOJI_LOG_MAPPING:
        level = EMOJI_LOG_MAPPING[emoji]
    else:
        level = default_level
    
    # Log the message
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(clean_message)