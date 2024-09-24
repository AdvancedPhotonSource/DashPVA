import numpy as np

x_start = 0
x_end = 1

y_start = 0
y_end = 1

def generate_raster_scan_positions(size=30): #TODO: x and y have different range. 
    x_positions = np.zeros(shape=(size,size), dtype=np.int8)
    y_positions = np.zeros(shape=(size,size), dtype=np.int8)

    x_positions[::2] += np.arange(size)
    x_positions[1::2] += np.arange(size-1,-1,-1)
    x_positions = x_positions.ravel()   

    y_positions+= np.arange(size)[:,np.newaxis]
    y_positions = y_positions.ravel()
    
    return np.array(x_positions), np.array(y_positions)

x_pos, y_pos = generate_raster_scan_positions()
xpos = ((x_end-x_start)/29.0) * x_pos + x_start
ypos = ((y_end-y_start)/29.0) * y_pos + y_start
print(ypos)
np.save('scan_plan_x.npy', xpos)
np.save('scan_plan_y.npy', ypos)
