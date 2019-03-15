import logging
import sys
from datetime import datetime


class DispatchingFormatter:

    def __init__(self, formatters, default_formatter):
        self._formatters = formatters
        self._default_formatter = default_formatter

    def format(self, record):
        formatter = self._formatters.get(record.name, self._default_formatter)
        return formatter.format(record)


def logger_initialization(log_level):

    # logLevel = ['DEBUG', 'INFO', 'ERROR']
    # no logLevel, default to INFO
    if not log_level:
        logging.getLogger().setLevel(getattr(logging, 'INFO'))
    else:
        log_level = log_level.upper()
        logging.getLogger().setLevel(getattr(logging, log_level))

    # not only log to a file but to stdout
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    handler_dictionary = {
        'regular': logging.Formatter("%(message)s"),
        'regular.line': logging.Formatter("%(message)s\n"),
        'line.regular': logging.Formatter("\n%(message)s"),
        'tab.regular': logging.Formatter("\t%(message)s"),
        'tab.tab.regular': logging.Formatter("\t\t%(message)s"),
        'tab.regular.line': logging.Formatter("\t%(message)s\n"),
        'tab.tab.regular.line': logging.Formatter("\t\t%(message)s\n"),
        'line.tab.regular': logging.Formatter("\n\t%(message)s"),
        'regular.time': logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'regular.time.line': logging.Formatter("%(asctime)s - %(message)s\n", "%Y-%m-%d %H:%M:%S"),
        'line.regular.time': logging.Formatter("\n%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'line.regular.time.line': logging.Formatter("\n%(asctime)s - %(message)s\n", "%Y-%m-%d %H:%M:%S"),
        'tab.regular.time': logging.Formatter("\t%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'tab.tab.regular.time': logging.Formatter("\t\t%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'tab.regular.time.line': logging.Formatter("\t%(asctime)s - %(message)s\n", "%Y-%m-%d %H:%M:%S"),
        'tab.tab.regular.time.line': logging.Formatter("\t\t%(asctime)s - %(message)s\n", "%Y-%m-%d %H:%M:%S"),
        'line.tab.regular.time': logging.Formatter("\n\t%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'debug.time': logging.Formatter("%(asctime)s - %(funcName)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'debug.time.line': logging.Formatter("%(asctime)s - %(funcName)s - %(message)s\n", "%Y-%m-%d %H:%M:%S"),
        'line.debug.time': logging.Formatter("\n%(asctime)s - %(funcName)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'tab.debug.time': logging.Formatter("\t%(asctime)s - %(funcName)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'tab.tab.debug.time': logging.Formatter("\t\t%(asctime)s - %(funcName)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'tab.debug.time.line': logging.Formatter("\t%(asctime)s - %(funcName)s - %(message)s\n", "%Y-%m-%d %H:%M:%S"),
        'tab.tab.debug.time.line': logging.Formatter("\t\t%(asctime)s - %(funcName)s - %(message)s\n",
                                                     "%Y-%m-%d %H:%M:%S"),
        'line.tab.debug.time': logging.Formatter("\n\t%(asctime)s - %(funcName)s - %(message)s", "%Y-%m-%d %H:%M:%S"),
        'line.tab.tab.debug.time': logging.Formatter("\n\t\t%(asctime)s - %(funcName)s - %(message)s",
                                                     "%Y-%m-%d %H:%M:%S"),
    }

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(DispatchingFormatter(handler_dictionary, logging.Formatter('%(message)s')))
    logging.getLogger().addHandler(handler)

    # create the logging file handler
    file_name = 'logfile_' + datetime.now().strftime('%Y%m%d%H%M%S') + '.log'
    fh = logging.FileHandler(file_name)
    fh.setFormatter(DispatchingFormatter(handler_dictionary, logging.Formatter('%(message)s')))

    msg = 'logfile = {0}'.format(file_name)
    logging.getLogger('line.regular.time.line').info(msg)

    # add handler to logger object
    logging.getLogger().addHandler(fh)
