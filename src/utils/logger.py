"""
Centralized Logging Utility
==========================

A unified logging system for all pipelines to ensure consistency
and reduce code duplication.
"""

import logging
import os
from typing import Optional


class PipelineLogger:
    """Centralized logger for all pipelines"""
    
    _loggers = {}  # Cache for loggers to prevent duplicates
    
    @classmethod
    def get_logger(cls, pipeline_name: str, log_file: Optional[str] = None) -> logging.Logger:
        """
        Get or create a logger for a specific pipeline.
        
        Args:
            pipeline_name: Name of the pipeline (e.g., 'data_ingestion', 'feature_store')
            log_file: Optional custom log file path
            
        Returns:
            Configured logger instance
        """
        # Return cached logger if it exists
        if pipeline_name in cls._loggers:
            return cls._loggers[pipeline_name]
        
        # Create new logger
        if log_file is None:
            log_file = f'logs/{pipeline_name}.log'
        
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create logger
        logger = logging.getLogger(pipeline_name)
        logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not logger.handlers:
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # File handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Cache the logger
        cls._loggers[pipeline_name] = logger
        
        return logger


def get_logger(pipeline_name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Convenience function to get a logger.
    
    Args:
        pipeline_name: Name of the pipeline
        log_file: Optional custom log file path
        
    Returns:
        Configured logger instance
    """
    return PipelineLogger.get_logger(pipeline_name, log_file)


# Common pipeline names for consistency
PIPELINE_NAMES = {
    'DATA_INGESTION': 'data_ingestion',
    'DATA_VALIDATION': 'data_validation', 
    'DATA_PREPARATION': 'data_preparation',
    'DATA_TRANSFORMATION': 'data_transformation_storage',
    'FEATURE_STORE': 'feature_store',
    'RAW_DATA_STORAGE': 'raw_data_storage',
    'DATA_VERSIONING': 'data_versioning',
    'BUILD_MODEL': 'build_model'
}
