import logging
logger = logging.getLogger("main_logger")
try:
    from IPython.core import debugger
    debug = debugger.Pdb().set_trace
except Exception as e:
    debug = lambda:None