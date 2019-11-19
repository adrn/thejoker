# Standard library
import logging

# Third-party
from astropy.logger import StreamHandler

__all__ = ['logger']

class JokerHandler(StreamHandler):
    def emit(self, record):
        record.origin = 'thejoker'
        super().emit(record)


class JokerLogger(logging.getLoggerClass()):
    def _set_defaults(self):
        """Reset logger to its initial state"""

        # Remove all previous handlers
        for handler in self.handlers[:]:
            self.removeHandler(handler)

        # Set default level
        self.setLevel(logging.INFO)

        # Set up the stdout handler
        sh = JokerHandler()
        self.addHandler(sh)


logging.setLoggerClass(JokerLogger)
logger = logging.getLogger('thejoker')
logger._set_defaults()
