import time
import pvaccess as pva


# try this on xsd6idb
class MyMonitor:
    def __init__(self):
        self.count = 0
        self.queue = []

    def monitor(self, pv):
        self.queue.append(pv['value'])

def main():
    # c = pva.Channel('6idkepco:LS336:TC1:IN4', pva.CA) # temperature sensor. can't get 10 Hz, highest 5 Hz
    
    c = pva.Channel('6idb1:m19.RBV', pva.CA) # able to get full 10 Hz when motor is moving
    m = MyMonitor()
    c.monitor(m.monitor)
    print('starting monitoring every 1 second')
    time_count = 0
    queue_len = 0
    while True:
        time.sleep(1)
        time_count += 1
        
        print(f'Received: {len(m.queue)}')
        print(f'avg PVs received per sec: {len(m.queue)/time_count}')
        print(f'PVs received during this sec: {len(m.queue) - queue_len}')
        queue_len = len(m.queue)

if __name__ == '__main__':
    main()
