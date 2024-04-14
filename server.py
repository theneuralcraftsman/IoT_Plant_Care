import socket
import time
from threading import Timer

import Adafruit_DHT
import time
from picamera import PiCamera

DHT_SENSOR = Adafruit_DHT.DHT11
DHT_PIN = 21

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 5000))
s.listen(5)
print('Server is now running.')

camera = PiCamera()

def background_controller(clientsocket):
	while True:
		user_input = clientsocket.recv(1024).decode("utf-8")
		if user_input.strip() == '1':
			message = get_data()
			print(message)
			clientsocket.send(bytes(message, "utf-8"))
			send_image(clientsocket, "test.jpg")
			clientsocket.send(b"end")
		else:
			print("issue")
	clientsocket.send(b"end")

def get_data():
	humidity, temperature = Adafruit_DHT.read(DHT_SENSOR, DHT_PIN)
	if humidity is not None and temperature is not None:
		return "Temp={0:0.1f}C Humidity={1:0.1f}%".format(temperature, humidity)
	else:
		return "Sensor failure"
	
def send_image(clientsocket, filename):
	time.sleep(5)
	#print("Capuring...")
	#camera.capture("test.jpg")
	with open(filename, "rb") as f:
		image_data = f.read()
		clientsocket.send(image_data)

while True:
	clientsocket, address = s.accept()
	print(f"Connection from {address} has been established.")
	background_controller(clientsocket)
	clientsocket.close()

