# HyTeG Operator Generator
# Copyright (C) 2024  HyTeG Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
import time
import sys
from logging.handlers import TimedRotatingFileHandler
import inspect

from hog.exception import HOGException

FORMATTER = logging.Formatter(
    "[%(processName)s] — [%(asctime)s] — [%(filename)32s] — "
    "[line %(lineno)4d] - [%(levelname)8s] %(message)s"
)
LOG_FILE = "hog.log"


def _get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def _get_file_handler():
    file_handler = TimedRotatingFileHandler(LOG_FILE, when="midnight")
    file_handler.setFormatter(FORMATTER)
    return file_handler


def get_logger(level=logging.INFO):
    """Use this function to obtain a python logger instance."""
    logger = logging.getLogger()
    logger.setLevel(level)
    if not logger.hasHandlers():
        logger.addHandler(_get_console_handler())
        # logger.addHandler(_get_file_handler())
    logger.propagate = False
    return logger


class TimedLogger:

    LOG_LEVEL = logging.INFO

    _CURRENT_DEPTH = 0

    @staticmethod
    def set_log_level(log_level: int) -> None:
        TimedLogger.LOG_LEVEL = log_level

    def __init__(self, msg: str, level: int = logging.INFO) -> None:
        self._msg = msg
        self._is_timer_running = False
        self._start_time = 0.0
        self._logger = get_logger(TimedLogger.LOG_LEVEL)
        self._level = level

    def log(self) -> None:
        self._log(self._msg)

    def _log(self, msg: str) -> None:
        indent = "".join(["  " for i in range(TimedLogger._CURRENT_DEPTH)])
        if sys.version_info[1] > 7:
            # stacklevel argument added in python 3.8
            self._logger.log(self._level, indent + msg, stacklevel=3)
        else:
            self._logger.log(self._level, indent + msg)

    def _start_timer(self) -> None:
        if self._is_timer_running:
            raise HOGException("Timer is already running.")
        self._start_time = time.time()

    def _stop_timer(self) -> float:
        delta = time.time() - self._start_time
        return delta

    def __enter__(self):
        TimedLogger._CURRENT_DEPTH += 1
        self._log(self._msg + " ...")
        self._start_timer()
        return self

    def __exit__(self, *exc_info: object) -> None:
        delta = self._stop_timer()
        if delta < 1e-2:
            delta *= 1000
            self._log(self._msg + f" ... done (took {delta:.0f} ms)")
        elif delta >= 60:
            delta /= 60
            self._log(self._msg + f" ... done (took {delta:.2f} min)")
        else:
            self._log(self._msg + f" ... done (took {delta:.2f} s)")
        TimedLogger._CURRENT_DEPTH -= 1
