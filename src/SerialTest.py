import serial
import time

arduino = serial.Serial("COM5", 9600, timeout=1)

time.sleep(2)  # Arduino often resets when serial opens

while True:
    line = arduino.readline().decode(errors="ignore").strip()

    if line.startswith("DIST:"):
        try:
            distance = float(line.split(":")[1])

            print(f"Distance: {distance:.1f} cm")

            if distance < 20:
                command = "RED"
            elif distance < 50:
                command = "YELLOW"
            else:
                command = "GREEN"

            arduino.write((command + "\n").encode())

        except ValueError:
            pass