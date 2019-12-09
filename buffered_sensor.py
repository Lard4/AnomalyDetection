import time
import pandas

read_file = "out.csv"
x_axis = "vel_x"
y_axis = "vel_y"
pub_file = open("pub.csv", 'x')
INIT_BUFFER = "vel_x, vel_y" + '\n'
buffer = ""
count = 0
prev_length = 0

while True:
    df = pandas.read_csv(read_file)

    # dont double count
    if len(df) == prev_length:
        pass

    buffer += df[x_axis] + ", " + df[y_axis] + '\n'
    count += 1
    prev_length += 1

    if count == 50:
        print(buffer, file=pub_file)
        print("WROTE " + str(count) + " LINES at time=" + time.time())
        buffer = INIT_BUFFER
        count = 0
