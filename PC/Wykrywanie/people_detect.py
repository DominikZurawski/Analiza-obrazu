#!/usr/bin/env python
# -*- coding: utf-8 -*- 
'''
# USAGE
# czytanie i zapis wideo:
# python people_detect.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 --output output/output_01.avi
#
# Strumien z kamery i zapis na dysku:
# python people_detect.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --output output/webcam_output.avi
#
# Odbior obrazu z pliku typu potok - fifo264
	python people_detect.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input ~/Desktop/Odbior\ wideo/fifo264

'''
# import niezbędnych bibliotek
from funkcje.centroidtracker import CentroidTracker
from funkcje.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
#from funkcje.swiatlo import swiatla
from funkcje.client import sendTCP
#from funkcje.mysz import click_and_crop
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import sys
sys.path.insert(0, '/usr/local/lib/python3.5/dist-packages')
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import socket
from array import *

refPt = [(90,90),(340,290),(0,90),(0,290),(500,90),(500,290)]
cropping = False
SYGNAL=False
z=0

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
    global refPt, cropping
       
    if param == "p":
        p = 0
	    # if the left mouse button was clicked, record the starting
	    # (x, y) coordinates and indicate that cropping is being
	    # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt[p] =(x, y)
            cropping = True
             
	    # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
		    # record the ending (x, y) coordinates and indicate that
		    # the cropping operation is finished
            refPt[p+1] = (x, y)
            cropping = False
            print("Przejscie ",refPt) 
            
    elif param == "1":
        p = 2    
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt[p] =(x, y)
            cropping = False
            print("Przejscie ",refPt)             
    elif param == "2":
        p = 3    
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt[p] =(x, y)
            cropping = False 
            print("Przejscie ",refPt)                       
    elif param == "3":
        p = 4
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt[p] =(x, y)
            cropping = False 
            print("Przejscie ",refPt)                                           
    elif param == "4":
        p = 5    
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt[p] =(x, y)
            cropping = False 
            print("Przejscie ",refPt)             
    elif param == "o":
        p = 5    
        return True 
        
def classify_frame(net, inputQueue, outputQueue):
    # keep looping
    while True:
        # check to see if there is a frame in our input queue
        if not inputQueue.empty():
            # grab the frame from the input queue, resize it, and
            # construct a blob from it
            frame = inputQueue.get()
            frame = cv2.resize(frame, (400, 400))
            blob = cv2.dnn.blobFromImage(frame, 0.007843,(400, 400), 127.5)
 
            # set the blob as input to our deep learning object
            # detector and obtain the detections
            net.setInput(blob)
            detections = net.forward()
 
            # write the detections to the output queue
            outputQueue.put(detections)

# Argumnety dołączane z linii poleceń
ap = argparse.ArgumentParser()		
ap.add_argument("-p", "--prototxt", required=True,
	help="ścieżka do pliku prototxt modelu Caffe")
ap.add_argument("-m", "--model", required=True,
	help="ścieżka do modelu Caffe")
ap.add_argument("-i", "--input", type=str,
	help="opcjonalna ścieżka do pliku wideo")
ap.add_argument("-o", "--output", type=str,
	help="opcjonalna ścieżka do zapisu wyniku wideo")
ap.add_argument("-c", "--confidence", type=float, default=0.1,
	help="minimalne prawdopodobieństwo filtrowania słabych detekcji")
ap.add_argument("-s", "--skip-frames", type=int, default=50, #30
	help="# ilość pominiętych ramek pomiędzy wykryciami")
args = vars(ap.parse_args())

# inicjalizacja listy etykiet klas modelu MobilNetSSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# ładowanie modelu z dysku
print("[INFO] ładowanie modelu...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# inicjalizacja kolejki przychodzących ramek i listy detekcji zwróconych przez proces potomny
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None
print("[INFO] starting process...")
p = Process(target=classify_frame, args=(net, inputQueue,outputQueue,))
p.daemon = True
p.start()

# inicjalizacja kamery w przypadku braku załączonego pliku wideo
if not args.get("input", False):
	print("[INFO] startowanie video stream...")
#	#vs = VideoStream(usePiCamera=True).start()
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
    
# ładowanie pliku wideo
else:
	print("[INFO] otwieranie pliku wideo...")
	vs = cv2.VideoCapture(args[r"input"])

writer = None		# pisarz wideo 
W = None		# inicjalizacja wymiarów ramki
H = None

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)		#tworzenie centroid tracker
trackers = []							#lista do przechowywania trackerów dlib
trackableObjects = {}						#słownik do mapowania kazdego inkalnego obiektu ID

totalFrames = 0				#całkowita liczba przetworzonych ramek
fps = FPS().start()	 		#uruchomienie liczenia statystyki przepustowości klatek na sekunde
pozition = {}
initBB = None

while True:
    start = time.time()

    #pętla na ramki ze strumienia wideo
    frame = vs.read()		
    frame = frame[1] if args.get("input", False) else frame		#przechwycenie ramki z wejścia wideo

    if args["input"] is not None and frame is None:			#obsługa końca pliku wideo
        break

    frame = imutils.resize(frame, width=500)		#zmiana ramki do szerokości 500 pikseli(mniej to szybciej)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)	#klatka od BGR do RGB dla dlib

    if W is None or H is None:				# ustawienie wymiarów ramki, gdy puste
        (H, W) = frame.shape[:2]

    if args["output"] is not None and writer is None:	#ustaw Writer jesli bez zapisu wideo 
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)
    Przejscie = "Brak"
    MSG_P = 'B'
    Chodnik = "Brak"
    MSG_C = 'B'
    status = "Czekam na sygnal"		#ustawienie bierzącego statusu oraz listy "prostokątów" 
    rects = []	
     
    if status == "Czekam na sygnal":
       Sygnal = False
       sendTCP(MSG_P+MSG_C)			


    if totalFrames % args["skip_frames"] == 0:	#sprawdzenie czy ominąć ramke
        status = "Wykrywanie"			#ustawienie statusu wraz z nowym zestawem modułów śledzenia obiektów
        trackers = []


# konwersja ramki na obiekt typu 'blob' i przeprowadzenie tego obiektu przez sieć, aby uzyskac rozpoznanie
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()
		
        for i in np.arange(0, detections.shape[2]):		# pętla nad wykryciami
            confidence = detections[0, 0, i, 2] 		#wyodrębnienie prawdopodobieństwa wykrycia związane z prognozą

            if confidence > args["confidence"]: 		#odfiltrowanie słabych detekcji wymagając minimalnej pewności
                idx = int(detections[0, 0, i, 1]) 		#wyodrębnienie etykiety klasy z listy detekcji

                if CLASSES[idx] != "person" and CLASSES[idx] != "bicycle": 	#ignorowanie nieprzydatnych wyników
                    continue

                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H]) 	#obliczenie współrzędnych obiektu
                (startX, startY, endX, endY) = box.astype("int")

		# konstrukcja obiektu prostokątnego 'dlib' ze wspólrzędnymi prostokąta ograniczającego 
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect) 					

                trackers.append(tracker) 	#dodanie trackera do listy, aby można go było wykorzystac podczas pominięcia ramek w przeciwnym wypadku wykorzystujemy obiekt liste trackerów (nie detektora obiektu), aby uzyskać wyższą przepustowość przetwarzania ramki
    else:
        for tracker in trackers:   	#pętla na liscie trackerow
            status = "Sledzenie" 

            tracker.update(rgb) 	#aktualizacja modułu śledzącego i pobranie zaktualizowanej pozycji
            pos = tracker.get_position()

            startX = int(pos.left())	# rozpakowanie polozenia obiektu
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            rects.append((startX, startY, endX, endY)) #dodanie wspólrzędnych prostokąta do listy prostokątów

	# rysuje pozimą linię na środlu, gdy obiekt przekroczy linię, zostaję określony kierunek poruszania się w górę lub w dół
	#cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
    x1 = refPt[0][0]
    y1 = refPt[0][1]
    x2 = refPt[1][0]
    y2 = refPt[1][1]
    dy1 = refPt[2][1]
    dy2 = refPt[3][1]
    dy3 = refPt[4][1]
    dy4 = refPt[5][1]
    d1 = np.array ([[0,dy1],[0,dy2],[x1,y2],[x1,y1]], np.int32) 	#droga1
    d2 = np.array ([[W,dy3],[W,dy4],[x2,y2],[x2,y1]], np.int32)   	#droga2
    
    cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 255), 2)  	#przejscie   
       
    cv2.polylines(frame, [d1],True,(255,0,0))         			#droga 1
    cv2.polylines(frame, [d2],True,(255,0,0))          			#droga 2

    # powiązanie starych centoitow obiektu z nowymi
    objects = ct.update(rects)

    # pętla nad sledzonymi obiektami
    for (objectID, centroid) in objects.items():
	# sprawdzenie czy istnieją możliwe do śledzenia obiekty dla bieżących ID
        to = trackableObjects.get(objectID, None)

	# jesli nie ma obiektu do sledzenia, stworzenie go
        if to is None:
            to = TrackableObject(objectID, centroid)
            

        else:  		#istnieje obiekt, który mozna uzyc do okreslenia kierunku # rożnica miedzy bieżącą współrzędna Y i srednią poprzednich, pozwala okreslić kierunek poruszania(ujemne w góre)
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            point = Point(centroid[0],centroid[1])
            polygon = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])
            przejscieGora = Polygon([(x1,y1),(x2,y1),(W,dy4),(W,0),(0,0),(0,dy1)])
            przejscieDol = Polygon([(x2,y1),(x2,y2),(W,dy4),(W,H),(0,H),(0,dy2)])

            if not to.counted:			#sprawdzenie czy obiekt został policzony
                if polygon.contains(point) : 	#czy na przejsciu                   
                    Przejscie = "Piesi"
                    MSG_P = 'P'
                elif przejscieGora.contains(point) or przejscieDol.contains(point):
                    Chodnik = "Piesi"
                    MSG_C = 'C'
		
        trackableObjects[objectID] = to		# przechowywanie obiektów mozliwych do sledzenie w słowniku
	
        if Przejscie == "Piesi":
            Sygnal = True
            sendTCP(MSG_P+MSG_C)
        elif Chodnik == "Piesi":
            sendTCP(MSG_P+MSG_C)
        elif status == "Czekam na sygnal":
            MSG_P = 'B'
            MSG_C = 'B'
            Sygnal = False
            sendTCP(MSG_P+MSG_C)
      
        text = "ID {}".format(objectID)		# rysowanie identyfikatora i srodka cieżkosci na ramie wyjsciowej
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.circle(frame, (centroid[0], centroid[1]), 3, (0, 255, 0), -1)

    info = [					# tuple z informacja do wyswietlenia na rame wideo
        ("Chodnik", Chodnik),
        ("Status", status),
        ("Przejscie",Przejscie)
    ]
	
    for (i, (k, v)) in enumerate(info):		# zapętlenie tupli i rysowanie ich na ramce
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        #print("			
	
    if writer is not None:			# sprawdzenie czy zapisac ramke na dysku
        writer.write(frame)
    
    key = cv2.waitKey(1) & 0xFF			#czekanie na akcje z klawiatury
        
    if key == ord("p"):
        #time.sleep(2.0)
        clone = frame.copy()
        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", click_and_crop,"p")
        #key = cv2.waitKey(1) & 0xFF        
    elif key == ord("1"):
        cv2.setMouseCallback("Frame", click_and_crop,"1")
    elif key == ord("2"):
        cv2.setMouseCallback("Frame", click_and_crop,"2")            
    elif key == ord("3"):
        cv2.setMouseCallback("Frame", click_and_crop,"3")                
    elif key == ord("4"):
        cv2.setMouseCallback("Frame", click_and_crop,"4") 
    elif key == ord("o"):
        cv2.setMouseCallback("Frame", click_and_crop,"o")           

         
    elif key == ord("q"):			# okreslenie przerwania programu klawisz 'q'
        break
    
    elif key == ord("z"):
        z=1
    
    #inp = input()
    if z==0:
        cv2.imshow("Frame", frame)		# wyswietlenie ramki wyjsciowej
        
    elif z==1:
        cv2.destroyAllWindows() 
        
   #elif input() == "z":
    #    z=0
     #   continue
    
    totalFrames += 1		# zwiększenie calkowitej liczby przetworzonych ramek i aktualizacja FPS
    fps.update()
    end = time.time()
fps.stop()			# zatrzymanie timera i wyswietlenie informacji FPS
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if writer is not None:			# sprawdzenie czy jest potrzeba zwolnienia wskaznika nagrywania wideo
	writer.release()
if not args.get("input", False):	# zatrzymanie strumienia wideo z kamery
	vs.stop()
else:					# zwolnienie wskaznika pliku wideo
	vs.release()
cv2.destroyAllWindows() 		# zamknięcie wszystkich okien
