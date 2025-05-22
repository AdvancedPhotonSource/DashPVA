import pvaccess as pva
import numpy as np
import time

previous_data = None
last_print_time = time.time()

total_intensities = []

import matplotlib.pyplot as plt


def monitor_callback(data):
    global previous_data, last_print_time
    
    current_time = time.time()
    # if current_time - last_print_time >= 5:  # Print only every 5 seconds
    if True:
        last_print_time = current_time
        
        print("\nReceived data:")
        # print(data)
        print(data['uniqueId'])
        # image_data = data['value'][0]['ubyteValue']
        # image_data = data['value'][0]['uintValue']
        # print(data.get())
        # print(data.getIntrospectionDict())
        # image_data = data['value'][0]['ushortValue']
        # print(data.has_key('uncompressedSize'))
        # print(data['compressedSize'], data['uncompressedSize'])
        # print(data.get())
        # total_intensities.append(np.sum(image_data))
        # print(np.size(image_data))
    
        # print(f"appending image total intensity {np.sum(image_data)}")

        # print(f"Image data length: {len(image_data)}")

        # metadata = {}
        # if 'attribute' in data:
        #     attributes = data['attribute']
        #     for attr in attributes:
        #         name = attr['name']
        #         value = attr['value'][0]['value']
        #         metadata[name] = value
                
        # print("\nMetadata:")
        # for channel, value in metadata.items():
        #     print(f"{channel} = {value}")
        # print(f"attributes diff = {metadata['RSM']['attributes_diff']}")

            
            # if previous_data is not None:
            #     if len(previous_data) != len(metadata):
            #         dicts_equal = False
            #     else:   
            #         for key, value in metadata.items():
            #             if key not in previous_data:
            #                 dicts_equal = False
            #                 print(key)
            #                 break
            #             if isinstance(value, np.ndarray):
            #                 arrs_equal = np.array_equal(value, previous_data[key])
            #                 if not arrs_equal:
            #                     dicts_equal = False
            #                     print(key)
            #                     break
            #             elif previous_data[key] != metadata[key]:
            #                 dicts_equal = False
            #                 print(key)
            #                 break
            #         else:
            #             dicts_equal = True

            #         print(f'Current equals Previous: {dicts_equal}')
             
            # previous_data = metadata
            # metadata["uniqueId"] = data["uniqueId"


        # print([metadata[f'PrimaryBeamDirection:AxisNumber{i}'] for i in range(1,4)])
        # primary_beam_directions = [metadata.get(f'PrimaryBeamDirection:AxisNumber{i}', None) for i in range(1,4)]
        # inplane_beam_direction = [metadata.get(f'PrimaryBeamDirection:AxisNumber{i}', None) for i in range(1,4)]
        # sample_surface_normal_direction = [metadata.get(f'SampleSurfaceNormalDirection:AxisNumber{i}', None) for i in range(1,4)]

        # print(primary_beam_directions, inplane_beam_direction, sample_surface_normal_direction)


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
        

# collector_channel = pva.Channel('processor:1:output', pva.PVA)
# collector_channel = pva.Channel("processor:10:analysis", pva.PVA)
# collector_channel = pva.Channel("collector:1:output", pva.PVA)
collector_channel = pva.Channel("collector2:1:output", pva.PVA)

collector_channel.subscribe("monitor", monitor_callback)
collector_channel.startMonitor()

try:
    while True:
        time.sleep(0.1)

except KeyboardInterrupt:
    collector_channel.stopMonitor()
