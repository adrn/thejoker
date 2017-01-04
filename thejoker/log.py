from __future__ import print_function

# Standard library
import logging

# Third-party
from astropy.extern.six import PY3
from astropy.logger import StreamHandler
from astropy.utils import find_current_module

__all__ = ['log']

Logger = logging.getLoggerClass()
class JokerLogger(Logger):

    def _set_defaults(self):
        """
        Reset logger to its initial state
        """

        # Remove all previous handlers
        for handler in self.handlers[:]:
            self.removeHandler(handler)

        # Set levels
        self.setLevel(logging.INFO)

        # Set up the stdout handler
        sh = StreamHandler()
        self.addHandler(sh)

    def makeRecord(self, name, level, pathname, lineno, msg, args, exc_info,
                   func=None, extra=None, sinfo=None):
        if extra is None:
            extra = {}
        if 'origin' not in extra:
            current_module = find_current_module(1, finddiff=[True, 'logging'])
            if current_module is not None:
                extra['origin'] = current_module.__name__
            else:
                extra['origin'] = 'unknown'

        if PY3:
            return Logger.makeRecord(self, name, level, pathname, lineno, msg,
                                     args, exc_info, func=func, extra=extra,
                                     sinfo=sinfo)
        else:
            return Logger.makeRecord(self, name, level, pathname, lineno, msg,
                                     args, exc_info, func=func, extra=extra)

logging.setLoggerClass(JokerLogger)
log = logging.getLogger('thejoker')
log._set_defaults()
