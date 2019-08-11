import cv2
import numpy as np


# Capturing the Video frames using cv2 from camera having id=0
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('face-detector.xml')

count = 0

face_data = []
dataset_path = './facedata/'

file_name = str(input("Enter the name of the person : "))

while True:
	# Reading the captured image. This method returns 2 values- boolean value and a frame
	ret, frame = cap.read()

	# Boolean Value of cap.read is stored in 'ret' which when false shows that image is not captured properly.
	# Reasons could be webcam is not started, or webcam is defected.
	if ret == False:
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	# detectMultiScale(frame,ScalingFactor,NoOfNeighbours)
	faces = face_cascade.detectMultiScale(frame,1.3,5)

	# Sorting the faces from largest to smallest based upon the area of the frame captured
	# We can find the area by multiplying width and the height (w*h)
	# This can be done by taking width = faces[2] and height = faces[3]
	# Area = faces[2]*faces[3]
	faces = sorted(faces,key=lambda f:f[2]*f[3],reverse=True)

	for face in faces[:1]:
		# rectangle(frame, startingCoordinate, endingCoordinate, Color, thickness)
		(x,y,w,h) = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),5)

		# Extract (Crop out the required face) : Region of Interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		# Store every 10th face
		count += 1
		if(count%10==0):
			cv2.imshow("Face Section",face_section)
			face_data.append(face_section)
			print(len(face_data))

	cv2.imshow("Frame",frame)


	#cv2.imshow("Gray Frame",gray_frame)

	# Wait for user input - q, then you will stop the loop
	# We understand the key_pressed by taking AND operation of "cv2.waitKey(1)" and "0xFF"
	# "cv2.waitKey(1)" gives us a 32bit number and "0xFF" is an 8 bit numberof all ones.
	# Taking AND operation of them gives us the last 8 bits of "cv2.waitKey(1)" which is basically a 8 Bit ASCII number
	key_pressed = cv2.waitKey(1) & 0xFF

	# In Python, ord(number/character/symbol) gives us 8 Bit ASCII number
	if key_pressed == ord('q'):
		break


# Convert out face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

# Save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully saved at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()



"""
scaleFactor - Parameter specifying how much the image size is reduced at each image scale.
Basically the scale factor is used to create your scale pyramid. ScaleFactor=1.3 means shrinking the original image by 30%
scaleFactor=1.05 means shrinking by 5%.

minNeighbours - Parameter specifying how many neighbours each candidate rectangle should have to retain it.
This parameter will affect the quality of the detected faces. Higher value results in less detections but more accuracy.
"""