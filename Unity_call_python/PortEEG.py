from psychopy import parallel
import time
port = parallel.ParallelPort(address=0x0378)
def SD():
    port.setData(0) 
    time.sleep(1)
    port.setData(192)



