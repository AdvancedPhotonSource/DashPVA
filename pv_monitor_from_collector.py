import pvaccess as pva
import numpy as np
previous_data = None

def monitor_callback(data):
    global previous_data
    
    print("\nReceived data:")
    image_data = data['value'][0]['ubyteValue']
    # image_data = data['value'][0]['uintValue']
    
    print(f"Image data length: {len(image_data)}")
    
    metadata = {}
    if 'attribute' in data:
        attributes = data['attribute']
        for attr in attributes:
            name = attr['name']
            value = attr['value']
            metadata[name] = value
    
    print("\nMetadata:")
    for channel, value in metadata.items():
        print(f"{channel}: {value}")
    
    if previous_data is not None:
        # Compare image data
        previous_image_data = previous_data['value'][0]['ubyteValue']
        # previous_image_data = previous_data['value'][0]['uintValue']
        images_equal = (image_data == previous_image_data).all()
        print("mean square err:" , np.sum(np.sqrt(previous_image_data**2-image_data**2)))
        print(f"Images are equal: {images_equal}")
        
    else:
        print("No previous data to compare.")
    if previous_data == None:    
        previous_data = data
        print('data recorded!')

collector_channel = pva.Channel("pvapy:image", pva.PVA)
# collector_channel = pva.Channel("collector:1:output", pva.PVA)
# collector_channel = pva.Channel("dp-ADSim:Pva1:Image", pva.PVA)
collector_channel.subscribe("monitor", monitor_callback)
collector_channel.startMonitor()

import time
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    collector_channel.stopMonitor()
