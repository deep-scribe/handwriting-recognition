import serial

ser = serial.Serial('/dev/tty.usbmodem621', 115200)

while True:
	print ser.readline();