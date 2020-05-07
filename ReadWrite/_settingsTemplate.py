###########################################################
# Populate the following variables with your desired values.
# When done editing this file, rename it to _settings.py and leave it in the same directory. 

### local directory to save data.  
# This is relevant to the decorator called _backupDFs in ReadWrite.  As downloading data from a remote server is typically a bottlebeck, this decorator allows data to be automatically stored locally on your computer.  This variable is the filepath where this data is saved.  Make sure to create the directory and python will not make it for you.  I leave it as an empty string as it personal to each user.  Also, an empty string will cause the data to be redownloaded each time should you not wish to save the data locally.

localDirToSaveData=r''  #the r before the string is required for windows's file paths.  For windows, include two backward slashes at the end of the filepath.  E.g. for windows: localDirToSaveData=r'c:\DownloadedData\\'.  (Linux doesn't require this).

