'''
Script for logging progress in generation pipelien
'''

import logging
import pathlib 

def custom_logging(name, logfile_name=None, logfile_dir=None):
    # Initialize main logger object
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)  # Set main logger to "info" level

    # Formatter for handlers, including timestamp
    formatter = logging.Formatter('%(asctime)s, %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Add handler for logging to a text file
    if logfile_name and logfile_dir is not None:
        # Define logfile path
        logfile_path = logfile_dir / f"{logfile_name}.txt"

        # Initialize logger (a = append to an existing file)
        fileHandler = logging.FileHandler(filename=logfile_path, mode="a")

        # Set level of logger to log INFO
        fileHandler.setLevel(logging.INFO)

        # Set the formatter for the file handler with timestamp
        fileHandler.setFormatter(formatter)
        
        logger.addHandler(fileHandler)

    return logger