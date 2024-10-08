
import os as _os
from pathlib import Path as _Path
from psutil import cpu_percent as _cpu_percent

## TODO This sublib seems like it overlaps heavily with pythonMisc.  I should probably merges these two

def check_cpu_load(time_to_average_over=1e-1):
    """ Returns a percentage of total system CPU usage, averaged over time specified. """
    return _cpu_percent(time_to_average_over)


# def processFileName(filename):
#     """
#     Extracts teh base name, directory name, file extension, and file root from
#     a filename.

#     Parameters
#     ----------
#     filename : str
#         filename with directory information.

#     Returns
#     -------
#     baseName : str
#         The basename of the file (filename without directory)
#     dirName : str
#         Directory name where the file is located
#     fileExt : str
#         File extension
#     fileRoot : str
#         Filename without the extension

#     Examples
#     --------
#     Example 1::
        
#         filename='/media/john/T7-Blue/asdf/awesome_data.csv'
#         out=processFileName(filename)
#         print(out)
#         baseName,dirName,fileExt,fileRoot=out
        
#     Example 2::
        
#         print(processFileName('asdf.123'))
        
#     Example 3::
        
#         print(processFileName('asdf/123'))
#     """
#     ## This functionality has largely been placed with pathlib.Path
#     # filepath = _Path(filename)
#     # name = filepath.name
#     # stem = filepath.stem
#     # parent = filepath.parent
#     # suffix = filepath.suffix
#     # parent_plus_stem = parent / stem
#     fileRoot,fileExt=_os.path.splitext(filename)
#     baseName=_os.path.basename(filename)
#     dirName=_os.path.dirname(filename)
    
#     return baseName,dirName,fileExt,fileRoot


def setPwd(password,system,username):
    """ 
    Encrypts password using keyring, a password management tool.  
    
    Use in conjunction with getPwd()
    
    Parameters
    ----------
    password : str
        ssh password for hbtep server
    system : str
        name of ssh hbtep server.  
    username : str
        name of login on ssh hbtep server.  
    
    NOTES
    -----
    this function also ONLY works on a work station where the OS-based 
    function keyring is installed and the password has already been set for 
    that user.  i'm using this on ubuntu.  not sure if it'll work on windows
    """
    import keyring
    keyring.set_password(system,username,password)
    
    
def getPwd(system,username):
    """ 
    Returns unencrypted password
    
    Parameters
    ----------
    password : str
        ssh password for hbtep server
    system : str
        name of ssh hbtep server.  
    username : str
        name of login on ssh hbtep server.  
        
    Returns
    -------
    : str
        ssh password 
    
    NOTES
    -----
    this function also ONLY works on a work station where the OS-based 
    function keyring is installed and the password has already been set for 
    that user.  i'm using this on ubuntu.  not sure if it'll work on windows
    """
    import keyring
    return str(keyring.get_password(system, username))


def check_operating_system(verbose=False):
    from sys import platform
    if verbose is True:
        print('your os is: ', platform)
    return platform


def playBeep(durationInS=1, freqInHz=440):
    """
    Play short beep
    
    Examples
    --------
    Example 1 ::
        
        playBeep(0.5)
        
    References
    ----------
    https://stackoverflow.com/questions/16573051/sound-alarm-when-code-finishes
    https://pythonin1minute.com/how-to-beep-in-python/
    """
    ## TODO.  look into using beepy: https://github.com/prabeshdhakal/beepy-v1
    platform = check_operating_system()
    if 'win' in platform.lower():
        import winsound
        winsound.Beep(freqInHz, int(durationInS*1e3))
    elif 'linux' in platform or 'darwin' in platform: #darwin = mac os x
        import os
        try:
            os.system('play -nq -t alsa synth {} sine {}'.format(durationInS, freqInHz))
        except:
            print('This command only works on linux and macosx. \n If play not found error provided, you need to install play.  In ubuntu, type: sudo apt install sox')
    else:
        print("\a") # generic OS alert noise # likely does not work in windows
        print("OS not recognized.  Instead playing generic OS noise.")
        
        
# def check_if_dir_exists(
#         dirpath,
#         create_dir_if_not_there=True,
#         ):
    
#     if type(dirpath) is str:
#         dirpath = _Path(dirpath)
#     if dirpath.is_file() is True:
#         raise Exception("This is a file, not a directory. ")
#     elif dirpath.is_file() is False and dirpath.is_dir() is False:
#         raise Exception("This is neither a file nor a directory.  ")
#     elif dirpath.exists() is True and dirpath.is_dir() is True:
#         """ This is a directory  """
#         return True
#     elif dirpath.exists() is False:
#         """ This is (very likely) a directory and definitely does not exist """
#         if create_dir_if_not_there is True:
#             dirpath.mkdir()
#         return False
#     else:
#         raise Exception("Issue with filetype encountered. ")
        

def checkAndCreateDir(directory):
    """
    Checks to see if a directory exists and creates it if not
    """
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
        

# def todaysDate(formatting="standard"):
#     """
#     References
#     ----------
#     https://www.programiz.com/python-programming/datetime/current-datetime
#     """
#     from datetime import date
#     today=date.today()
#     if formatting=="standard":
#         return today.strftime("%d/%m/%Y")
#     elif formatting=="underscore":
#         return today.strftime("%Y_%m_%d")


def listDirectoriesForAllModules():
    import sys
    a=sys.path
    for i in range(0,len(a)):
        print("%s" % a[i])
        
    return a


def returnPythonVersion():
    import sys
    return sys.version_info.major






