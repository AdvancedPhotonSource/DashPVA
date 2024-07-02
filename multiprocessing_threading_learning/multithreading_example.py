import threading
import os
import time

def worker():
    print(f"Worker thread ID: {threading.get_ident()}")
    time.sleep(120)  # Simulate work for 2 minutes

if __name__ == "__main__":
    num_threads = 3  # Number of threads to spawn
    threads = []

    print(f"Main thread ID: {threading.get_ident()}")

    for _ in range(num_threads):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
