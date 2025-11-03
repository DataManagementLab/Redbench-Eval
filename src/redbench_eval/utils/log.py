import logging

LOGGER = logging.getLogger()
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    LOGGER.addHandler(ch)


def log(line, debug=False):
    if debug:
        LOGGER.debug(line)
    else:
        LOGGER.info(line)
