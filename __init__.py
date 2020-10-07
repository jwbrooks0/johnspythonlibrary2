from ._version import __version__


try:
	# This file does not exist by default.  Rename the template file and fill in the appropriate variables within.
    from . import Hbtep
except ModuleNotFoundError:
    print(Warning('MDS plus not installed. \nUntil you do, the Hbtep sub-library will be disabled.'))
except ImportError:
 	print(Warning('Warning: Hbtep/Get/_settings.py file not found. \nHave you modified the template file yet in Hbtep/Get/? \nUntil you do, the Hbtep sub-library will be disabled.'))


from . import Plot
from . import Process
from . import ReadWrite
from . import Image
from . import Networking
from . import OS
from . import PythonMisc
from . import OptomizeCode