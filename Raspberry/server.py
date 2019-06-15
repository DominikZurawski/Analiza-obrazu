import socket
from time import sleep

BUFFER_SIZE =10 	 #normally 1024, but we want fast respone
def serwer():
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	address = ('',5005)
	s.bind(address)
	s.listen(1)

	(conn, addr)= s.accept()
	print ('conetion address:', addr)
	
	while True:
		data = conn.recv(BUFFER_SIZE).decode()			
		if not data: break
		else:
			print ('received data:',data)
			conn.send(data) #echo
			conn.close()
			return data
	conn.close()
	
	
def decyzja():
	data = serwer()
	if data == u'BC':
		return 'Chodnik'
	elif data == u'PC':
		return 'Kontynuuj'
	elif data == u'PB':
		return 'Uwaga'
	elif data == u'BB':	
		return 'Brak'

