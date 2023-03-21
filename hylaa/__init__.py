'''This file defines which files to import when 'from hylaa import *' is used'''

__all__ = []

from . import _version
__version__ = _version.get_versions()['version']
__author__ = "Stanley Bak"
__copyright__ = "Copyright 2023"
__credits__ = [
   "Stanley Bak",
   "Hoang-Dung Tran",
   "Max von Hippel" 
]
__license__ = "GPLv3"
__maintainer__ = "Stanley Bak"
__email__ = "bak2007-DONTSENDMEEMAIL@gmail.com"
__status__ = "Prototype"