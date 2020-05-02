This is my personal python library and contains functions that I've developed, borrowed, or wrapped for my personal use. 


### Installing
Follow these steps to install this library.  Additional details are below.

1. Make sure the [MDSplus](http://www.mdsplus.org/index.php/Documentation:Users:MDSobjects:Python) library is installed and in the python path.
2. Clone this (johnspythonlibarary2) library.  Make sure its directory is in the PYTHONPATH environment variable.
3. (Optional) To use the Hbtep sublibrary, you must copy Readwrite/_settingsTemplate.py, save it as ReadWrite/_settings.py, and fill in the variables.
4. (Optional) To store (mostly Hbtep) data locally on your machine, copy Hbtep/Get/_settingsTemplate.py, save it as Hbtep/Get/_settings.py, and fill in the variables.


### Installing MDSplus on Ubuntu

Note that these are the steps that I used.  Check the website to make sure you use the latest and most relevant commands.

##### 1. Download and install MDSplus as discussed [here](http://www.mdsplus.org/index.php/Latest_Ubuntu/Debian_Packages)

###### a. Importing the signing key
apt-get -y install curl gnupg && curl -fsSL http://www.mdsplus.org/dist/mdsplus.gpg.key | apt-key add -

###### b. Enabling MDSplus Ubuntu Repository
sudo sh -c "echo 'deb http://www.mdsplus.org/dist/Ubuntu18/repo MDSplus stable' > /etc/apt/sources.list.d/mdsplus.list"

###### c. Update Repository
sudo apt-get update

###### d. Performing the installation
sudo apt-get install mdsplus-python

###### e. Updating your installation
sudo apt-get upgrade "mdsplus*"

##### 2. Install MDSplus python library
Follow the steps here: [Ref. 1](http://www.mdsplus.org/index.php/Documentation:Users:MDSobjects:Python) 

If using virtualenv, these steps need to be modified as shown here: [Ref. 2](https://h1ds.readthedocs.io/en/latest/intro/install.html)

Within your virtualenv, do the following:
*  mkdir $VIRTUAL_ENV/src
*  mkdir $VIRTUAL_ENV/src/python-mdsplus
*  cp -rp /usr/local/mdsplus/python/MDSplus $VIRTUAL_ENV/src/python-mdsplus
*  python $VIRTUAL_ENV/src/python-mdsplus/./setup.py install

