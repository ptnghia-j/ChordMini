"""
Standalone logging utilities for the ChordMini pipeline.
"""

import logging
import sys
import time

logger = logging.getLogger('ChordMini')
logger.setLevel(logging.INFO)
logger.propagate = False

class _Formatter(logging.Formatter):
    def format(self, record):
        ts = time.strftime('%m-%d %H:%M:%S', time.localtime(record.created))
        return f'I ChordMini {ts} {record.filename}:{record.lineno}] {super().format(record)}'

if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setLevel(logging.INFO)
    _h.setFormatter(_Formatter())
    logger.addHandler(_h)

def info(message):
    logger.info(message)

def warning(message):
    logger.warning(message)

def error(message):
    logger.error(message)

def debug(message):
    logger.debug(message)

def logging_verbosity(level):
    if level == 0:
        logger.setLevel(logging.WARNING)
    elif level == 1:
        logger.setLevel(logging.INFO)
    elif level >= 2:
        logger.setLevel(logging.DEBUG)
