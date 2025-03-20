import pvaccess as pva
import numpy as np
import time

previous_data = None
last_print_time = time.time()

total_intensities = []
metadata0_list = []

import matplotlib.pyplot as plt


def monitor_callback(data):
    global previous_data, last_print_time
    
    current_time = time.time()
    # if current_time - last_print_time >= 5:  # Print only every 5 seconds
    if True:
        last_print_time = current_time
        
        print("\nReceived data:")
        # print(data)
        # image_data = data['value'][0]['ubyteValue']
        # image_data = data['value'][0]['uintValue']
        image_data = data['value'][0]['ushortValue']
        print(data['dimension'])
        # print(data.get())
        # total_intensities.append(np.sum(image_data))
        # print(np.size(image_data))
    
        # print(f"appending image total intensity {np.sum(image_data)}")

        # print(f"Image data length: {len(image_data)}")

        metadata = {}
        if 'attribute' in data:
            attributes = data['attribute']
            for attr in attributes:
                name = attr['name']
                value = attr['value']
                metadata[name] = value
            metadata["uniqueId"] = data["uniqueId"]

        print("\nMetadata:")
        for channel, value in metadata.items():
            print(f"{channel}: {value}")
            if channel == "processor:1:analysis":
                print(f"appending {value[0]['value']}")
                metadata0_list.append(value[0]['value'])

        # if previous_data is not None:
        #     images_equal = (image_data == previous_data).all()
        #     print("mean square err:", np.sum(np.sqrt(previous_data**2 - image_data**2)))
        #     print(f"Images are equal: {images_equal}")

        # else:
        #     print("No previous data to compare.")
            
        if previous_data is None:
            # previous_data = data['value'][0]['ubyteValue']
            # previous_data = data['value'][0]['uintValue']
            # previous_data = data['value'][0]['ushortValue']
            print('data recorded!')
        

# collector_channel = pva.Channel('DetectorSetup:Name', pva.CA)
# collector_channel = pva.Channel("processor:1:analysis", pva.PVA)
collector_channel = pva.Channel("pvapy:image", pva.PVA)
collector_channel.subscribe("monitor", monitor_callback)
collector_channel.startMonitor()

try:
    while True:
        time.sleep(0.1)
        # print(f"{total_intensities}, {metadata0_list}")

except KeyboardInterrupt:
    collector_channel.stopMonitor()
