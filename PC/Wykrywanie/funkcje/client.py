#client
import socket
import sys

BUFFER_SIZE = 10
    
def sendTCP(MESSAGE):	
		try:
			with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:	
				s.connect(('192.168.0.100',5005))
				s.send(bytes(MESSAGE,"utf-8"))
				#s.send(MESSAGE.encode())
				data = s.recv(BUFFER_SIZE)
				s.close()
				#print ("received: " + data)
		except socket.error as msg:
			s.close()
			s = None
			#continue

