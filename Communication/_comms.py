

from threading import Thread as _Thread, Lock as _Lock
import socket as _socket
from time import sleep as _sleep


# %% file transfer

class scpData:
    """
    ssh and scp combination toolkit for file transfers
    
    Parameters
    ----------
    password : str
        password for ssh server
    address : str
        address for ssh server
    port : int
        port number for ssh server
    username : str
        username for ssh server
 
    Attributes
    ----------
    ssh : 'paramiko.SSHClient'
    scp : 'scp.SCPClient'
    
    Notes
    -----    
    
    References
    ----------
    https://gist.github.com/stonefury/06ab3531a1c30c3b998a
    https://github.com/jbardin/scp.py
    
    Examples
    --------
    
    """        
#    def __init__(self,password=getPwd(),username=_pref._HBT_SERVER_USERNAME,
#                 address=_pref._HBT_SERVER_ADDRESS,port=22):
                     
    def __init__(self,password,username,
                 address,port=22):
        from paramiko import SSHClient
        from scp import SCPClient
#        print('password=%s, username=%s' %(password,username))
        self.ssh = SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.connect(address, port=port, username=username, 
                         password=password)
        self.scp = SCPClient(self.ssh.get_transport())
        
    def downloadFile(self,remoteFilePath,localFilePath=''):
#        try:
        self.scp.get(remoteFilePath, localFilePath)
#        except:
#            print("%s not present.  skipping..." % remoteFilePath)
#            pass
        
    def uploadFile(self,localFilePath,remoteFilePath):
        self.scp.put(localFilePath,remoteFilePath)
        
    def uploadFolder(self,localDirPath,remoteDirPath):
        """ upload folder and all contents """
        self.scp.put(localDirPath,remoteDirPath,recursive=True)
        
    def closeConnection(self):
        self.scp.close();
        self.ssh.close();
        
    def __del__(self):
        """ upon deletion of object, close ssh/scp connections """
        self.closeConnection();



# %% TCP messaging and server
# This is super useful for passing real-time data between computers

def send_tcp_message(message, address=('127.0.0.1', 8000), term_char='\n'):

    # make sure the message is correctly terminated
    if message.encode()[-1] != term_char.encode()[-1]:
        message += term_char
    
    # Initialize a TCP client socket using SOCK_STREAM
    tcp_client = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    
    try:
        # Establish connection to TCP server and exchange data
        tcp_client.connect(address)
        tcp_client.sendall(message.encode())
    
        # (Optional) Read data from the TCP server and close the connection
        # received = tcp_client.recv(1024)
    finally:
        tcp_client.close()


class tcp_message_server:
    
    # class variables
    _lock = _Lock()  # a lock for the messages variable
    _messages = ''
    
    def __init__(self, address=('127.0.0.1', 8000), term_char='\n'):
        
        self._TERM_CHAR = term_char
        self._ADDRESS = address
        self.start()
        self.thread = _Thread(target = self.listen_receive_and_queue_messages)
        self.thread.start()
        
    def __enter__(self):
        return self

    def __exit__(self, a, b, c):
       self.close()
        
    def close(self):
        self._sock.close()
        
    def start(self):
        self._sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) 
        self._sock.bind(self._ADDRESS)
        # self._sock.settimeout(0.1)
        
    def listen_receive_and_queue_messages(self):
        while(True):            
            try:
                self._sock.listen()
                conn, addr = self._sock.accept()
                with self._lock:
                    self._messages += conn.recv(1024).decode()
            except _socket.timeout:
                pass
            except:
                raise SystemExit('closing thread')
            
    def read_messages(self):
        with self._lock:
            if self._TERM_CHAR in self._messages:
                index = self._messages.rindex(self._TERM_CHAR)
                data = self._messages[:index].split(self._TERM_CHAR)
                self._messages = self._messages[index + len(self._TERM_CHAR):]
                return data
            else:
                return []
            
            
# %% main
    
if __name__ == '__main__':
        
    with tcp_message_server(address=('127.0.0.1', 8000)) as serv:
        
        send_tcp_message('asdf')
        send_tcp_message('1234_wer')
        send_tcp_message("no way, that can't be!!   ")
        send_tcp_message("nfdhgfa!! \r")
        print(serv.read_messages())
        _sleep(0.01)
        print(serv.read_messages())
        print(serv.read_messages())
        
        
        
    
    