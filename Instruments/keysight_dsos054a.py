### This list of python libraries includes those needed to run johnspythonlibrary2.  It also includes a few libraries that I like to keep on hand.
# NOTE:  This does not include the MDSplus library.  See the attached readme for instructions on installing it. 
# Within your virtual environment, type the command: pip -r default_libraries.txt to install this list all in one go. 
# NOTE.  Make sure python3-dev is installed first.  sudo apt install python3-dev 
 
### python libraries
# spyder==3.3.6	# this is the final release of spyder v3.  I find spyder v4 to be buggy still, and I keep coming back to spyder v3.  
spyder			# spyder, most recent version
matplotlib		# plotting package
numpy			# data arrays and scientific tool library
pandas			# data structures, 1 dimentional
xarray			# data structures, 2+ dimensional
h5netcdf		# hdf5 file storage for xarray
ezodf			# open office related libraries
lxml			# processes processing XML and HTML.  required for ezodf
scipy			# scientific tool library
sklearn			# machine learning tools
Pillow			# image processing tools
# glob3			# unix-like filenames.  includes using wildcards * in filepaths.  # throws errors when pip install glob3 ...?
sh			# shell scripting library
deprecated		# has a decorator for flagging deprecated functions
# pathos		# multiprocessing sublibrary #TODO I think I'm not longer using this library
seaborn			# additional plotting functions, colormaps, etc
h5py			# loading matlab data
mat73			# loading matlab data (better)
sk-video
pyvisa			# used to talk with instruments over USB, GPIB, Ethernet, etc
pyserial 		# used to talk with instruments over serial (eg. RS232) connections.

### Spyder settings - I've also included a few of my favorite settings for my spyder environment.
# Consider reverting to the default settings first.  I've had issues where it pulls in an old spyder config file from outside the virtualenv.
# Note that these settings are for a mix of Spyder3 and Spyder4. 
### 
# Preferences->Editor->Display->Source Code->Indention characters->Tabulations
# Preferences->Editor->Advances->Edit template for new file->  (Then add numpy, pandas, mpl, etc)
# Preferences->IPython console->Graphics->Graphics backend->Backend->Automatic
# Preferences->Completion and linting->Code style and formatting->Enable code style linting
# Preferences->Completion and linting->Code style and formatting->Ignore the following errors or warnings-> W191, W293, W291, E501, E128
# Preferences->Completion and linting->Show automatic completions after
#    characters entered->1
# Preferences->Show automatic completions after
#    keyboard idle (ms)->100
# Preferences->Appearance->Syntax highlighting theme->Spyder Dark
### Misc.
# PYTHONPATH - remember to add your personal libraries

