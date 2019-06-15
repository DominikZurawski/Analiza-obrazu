import cv2

refPt = []
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
