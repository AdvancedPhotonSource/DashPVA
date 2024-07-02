import multiprocessing
import os
import time

def worker():
    print(f"Worker process ID: {os.getpid()}")
    time.sleep(120)  # Simulate work for 2 minutes

if __name__ == "__main__":
    num_processes = 3  # Number of processes to spawn
    processes = []

    print(f"Parent process ID: {os.getpid()}")

    for _ in range(num_processes):
        p = multiprocessing.Process(target=worker)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
