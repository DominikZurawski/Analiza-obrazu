import RPi.GPIO as GPIO
from time import sleep
from server import decyzja

#import threading
z_pieszy = 31
c_pieszy = 29

z_pojazd = 37
p_pojazd = 35
c_pojazd = 33

SYGNAL= False

def pieszy_zielone_przygotowanie():	
	GPIO.output(z_pojazd, GPIO.LOW) # stan niski,zolte
	GPIO.output(p_pojazd, GPIO.HIGH)
	GPIO.output(c_pojazd, GPIO.LOW)
	GPIO.output(c_pieszy, GPIO.HIGH)
	GPIO.output(z_pieszy, GPIO.LOW)		
	sleep(1)
	
def pieszy_zielone():	#ok
	GPIO.output(z_pojazd, GPIO.LOW)# czerwone
	GPIO.output(p_pojazd, GPIO.LOW)
	GPIO.output(c_pojazd, GPIO.HIGH)
	GPIO.output(z_pieszy, GPIO.HIGH)# stan wysoki,zielone
	GPIO.output(c_pieszy, GPIO.LOW)
	sleep(2)
	
def brak_pieszych_przygotowanie():
	GPIO.output(z_pojazd, GPIO.LOW)
	GPIO.output(p_pojazd, GPIO.HIGH) # pomaranczowy 
	GPIO.output(c_pojazd, GPIO.HIGH) # czerwony
	GPIO.output(z_pieszy, GPIO.LOW)
	GPIO.output(c_pieszy, GPIO.HIGH) # czerwone
	sleep(3)
	
def brak_pieszych():	#ok 
	GPIO.output(z_pojazd, GPIO.HIGH) # stan wysoki,zielone
	GPIO.output(p_pojazd, GPIO.LOW)
	GPIO.output(c_pojazd, GPIO.LOW)	
	GPIO.output(z_pieszy, GPIO.LOW)
	GPIO.output(c_pieszy, GPIO.HIGH) # czerwone
	sleep(2)
			
def swiatla(SYGNAL) :
	diody = (z_pieszy,c_pieszy,z_pojazd,p_pojazd,c_pojazd) #tuple
	flaga = 0
	GPIO.setmode(GPIO.BOARD)
	GPIO.setup(diody, GPIO.OUT)
	
	try:
		if SYGNAL == 'Chodnik':					
			pieszy_zielone_przygotowanie()			
			
			pieszy_zielone()
			sleep(5)
			while SYGNAL == 'Chodnik' or SYGNAL == 'Kontynuuj':
				pieszy_zielone()
				SYGNAL = decyzja()
				
				if SYGNAL == 'Uwaga':
					pieszy_zielone()
				
					while SYGNAL == 'Uwaga':
						pieszy_zielone()
						SYGNAL = decyzja()
						if SYGNAL == 'Brak':
							brak_pieszych_przygotowanie()
							break
				elif SYGNAL == 'Brak':
					brak_pieszych_przygotowanie()
					brak_pieszych()
					sleep(5)
					while SYGNAL == 'Brak':
						brak_pieszych()
						SYGNAL = decyzja()
						if SYGNAL == 'Chodnik': 
							pieszy_zielone_przygotowanie()
							break
		elif SYGNAL == 'Brak':
					brak_pieszych_przygotowanie()
					brak_pieszych()
					sleep(5)
					while SYGNAL == 'Brak':
						brak_pieszych()
						SYGNAL = decyzja()
						if SYGNAL == 'Chodnik': 
							pieszy_zielone_przygotowanie()
							break
		
	finally:
		GPIO.cleanup()

while 1:
	swiatla(decyzja())
