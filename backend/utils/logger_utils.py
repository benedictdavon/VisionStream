import logging

def setup_logger(level="INFO"):
    """
    Configure the global logger for VisionStream.
    Args:
        level (str): Logging level (DEBUG or INFO).
    """
    numeric_level = getattr(logging, level.upper(), None)
    if numeric_level not in (logging.DEBUG, logging.INFO):
        raise ValueError(f"Invalid log level: {level} (use DEBUG or INFO)")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger("VisionStream")
