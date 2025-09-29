import serial, time

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=None)
time.sleep(0.5)  # let the STM32 boot

for i in range(10):

    # Send ASCII '0' (0x30) to STM32
    ser.write(b'0')

    # Read back the echo or response (e.g. "Received 0\r\n")
    time1 = time.perf_counter_ns()
    resp1 = ser.read(1)
    time2 = time.perf_counter_ns()

    time3 = time.perf_counter_ns()
    resp2 = ser.read(1)
    time4 = time.perf_counter_ns()

    time5 = time.perf_counter_ns()
    resp3 = ser.read(1)
    time6 = time.perf_counter_ns()

    time7 = time.perf_counter_ns()
    resp4 = ser.read(1)
    time8 = time.perf_counter_ns()

    print(resp1, resp2, resp3, resp4)
    print(time2-time1, time4-time3, time6-time5, time8-time7)
    print(time4-time2, time6-time4, time8-time6)

ser.close()