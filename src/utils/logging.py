import logging

def get_logger(name, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    logger.addHandler(logging.StreamHandler())  # console
    if log_file:
        logger.addHandler(logging.FileHandler(log_file))
    return logger