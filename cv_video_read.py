import cv2

# Capturing the Video frames using cv2 from camera having id=0
cap = cv2.VideoCapture(0)

while True:
	# Reading the captured image. This method returns 2 values- boolean value and a frame
	ret, frame = cap.read()
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	# Boolean Value of cap.read is stored in 'ret' which when false shows that image is not captured properly.
	# Reasons could be webcam is not started, or webcam is defected.
	if ret == False:
		continue

	cv2.imshow("Video Frame",frame)
	cv2.imshow("Gray Frame",gray_frame)
	# Wait for user input - q, then you will stop the loop
	# We understand the key_pressed by taking AND operation of "cv2.waitKey(1)" and "0xFF"
	# "cv2.waitKey(1)" gives us a 32bit number and "0xFF" is an 8 bit numberof all ones.
	# Taking AND operation of them gives us the last 8 bits of "cv2.waitKey(1)" which is basically a 8 Bit ASCII number
	key_pressed = cv2.waitKey(1) & 0xFF

	# In Python, ord(number/character/symbol) gives us 8 Bit ASCII number
	if key_pressed == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()