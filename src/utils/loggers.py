"""Module to configure loggers."""

import logging
from logging.handlers import TimedRotatingFileHandler
from os.path import dirname, join, realpath
from utils.conf import Params

PARAMS_PATH = join(
    dirname(realpath(__file__)),
    "..",
    "..",
    "conf",
    "conf.yaml"
    )
logger_params = Params(PARAMS_PATH, 'Logger').get_dict_params()
logger_name = logger_params['name']
logger_file = logger_params['log_file']


def setup_loggers(level):
    """Instantiate and configure logger object."""

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    log_format = '%(asctime)s - %(name)s - %(levelname)s : %(message)s'
    formatter = logging.Formatter(log_format)

    logging.basicConfig(format=log_format)

    f_handler = TimedRotatingFileHandler(
        logger_file,
        when="D",
        interval=30,
        backupCount=5
    )
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(formatter)

    logger.addHandler(f_handler)

    return None
