#!/usr/bin/env python
"""
This script creates a PvaServer with multiple channels.
- Six axis channels update their "Position" field periodically.
- UBMatrix and Energy channels are static.
- Four additional static channels are created:
    - PrimaryBeamDirection
    - InplaneReferenceDirection
    - SampleSurfaceNormalDirection
    - DetectorSetup

Each record includes its own PV name (under the "Name" field) as part of its data.
"""

import time
import math
from pvaccess import ULONG, STRING, DOUBLE
from pvaccess import PvObject, PvaServer

# -------------------------------
# Define Axis PV record parameters
# -------------------------------

axis_records = {
    "6idb1:m28.RBV": {
        "AxisNumber": 1,
        "SpecMotorName": "Mu",
        "DirectionAxis": "x+",
        "Position": 0.0,
    },
    "6idb1:m17.RBV": {
        "AxisNumber": 2,
        "SpecMotorName": "Eta",
        "DirectionAxis": "z-",
        "Position": 10.74625,
    },
    "6idb1:m19.RBV": {
        "AxisNumber": 3,
        "SpecMotorName": "Chi",
        "DirectionAxis": "y+",
        "Position": 90.14,
    },
    "6idb1:m20.RBV": {
        "AxisNumber": 4,
        "SpecMotorName": "Phi",
        "DirectionAxis": "z-",
        "Position": 0.0,
    },
    "6idb1:m29.RBV": {
        "AxisNumber": 1,
        "SpecMotorName": "Nu",
        "DirectionAxis": "x+",
        "Position": 0.0,
    },
    "6idb1:m18.RBV": {
        "AxisNumber": 2,
        "SpecMotorName": "Delta",
        "DirectionAxis": "z-",
        "Position": 70.035125,
    },
}

# -------------------------------
# Define Static PV record parameters
# -------------------------------

# Existing static records:
ub_matrix_record = {
    "Value": [1, 0, 0, 0, 1, 0, 0, 0, 1]
}
energy_record = {
    "Value": 11.212  # keV
}

# Additional static records based on HKL configuration.
primary_beam_direction_record = {
    "AxisNumber1": 0,
    "AxisNumber2": 1,
    "AxisNumber3": 0
}
inplane_reference_direction_record = {
    "AxisNumber1": 0,
    "AxisNumber2": 1,
    "AxisNumber3": 0
}
sample_surface_normal_direction_record = {
    "AxisNumber1": 0,
    "AxisNumber2": 0,
    "AxisNumber3": 1
}
detector_setup_record = {
    "PixelDirection1": "z-",
    "PixelDirection2": "x+",
    "CenterChannelPixel": [237, 95],
    "Size": [83.764, 33.54],
    "Distance": 900.644,
    "Units": "mm"
}

# -------------------------------
# Helper functions to create PvObjects
# -------------------------------

def create_axis_pv(record_name, initial):
    """
    Creates a PvObject for an axis record.
    The type dictionary includes:
      - "Name": STRING
      - "AxisNumber": ULONG
      - "SpecMotorName": STRING
      - "DirectionAxis": STRING
      - "Position": DOUBLE
    The record_name is added to the data as the "Name" field.
    """
    typeDict = {
        "Name": STRING,
        "AxisNumber": ULONG,
        "SpecMotorName": STRING,
        "DirectionAxis": STRING,
        "Position": DOUBLE,
    }
    data = dict(initial)
    data["Name"] = record_name
    return PvObject(typeDict, data)

def create_static_pv(record_name, typeDict, initial):
    """
    Creates a PvObject for a static record.
    Adds the "Name" field with the record_name.
    """
    typeDict = dict(typeDict)  # make a copy so we don't modify the original
    typeDict["Name"] = STRING
    data = dict(initial)
    data["Name"] = record_name
    return PvObject(typeDict, data)

# -------------------------------
# Main server setup
# -------------------------------

if __name__ == '__main__':
    # Create a single PvaServer instance.
    server = PvaServer()

    # -------------------------------
    # Add Axis Records (Dynamic)
    # -------------------------------
    axis_pv_objects = {}
    for pvname, params in axis_records.items():
        pvObj = create_axis_pv(pvname, params)
        server.addRecord(pvname, pvObj)
        axis_pv_objects[pvname] = pvObj

    # -------------------------------
    # Add Static Records
    # -------------------------------
    # UBMatrix record.
    ubMatrixType = {"Value": [ULONG]}
    ubMatrixObj = create_static_pv("6idb:spec:UB_matrix", ubMatrixType, ub_matrix_record)
    server.addRecord("6idb:spec:UB_matrix", ubMatrixObj)

    # Energy record.
    energyType = {"Value": DOUBLE}
    energyObj = create_static_pv("6idb:spec:Energy", energyType, energy_record)
    server.addRecord("6idb:spec:Energy", energyObj)

    primaryBeamType = {
        "AxisNumber1": ULONG,
        "AxisNumber2": ULONG,
        "AxisNumber3": ULONG,
    }
    primaryBeamObj = create_static_pv("PrimaryBeamDirection", primaryBeamType, primary_beam_direction_record)
    server.addRecord("PrimaryBeamDirection", primaryBeamObj)

    inplaneReferenceType = {
        "AxisNumber1": ULONG,
        "AxisNumber2": ULONG,
        "AxisNumber3": ULONG,
    }
    inplaneReferenceObj = create_static_pv("InplaneReferenceDirection", inplaneReferenceType, inplane_reference_direction_record)
    server.addRecord("InplaneReferenceDirection", inplaneReferenceObj)

    sampleSurfaceNormalType = {
        "AxisNumber1": ULONG,
        "AxisNumber2": ULONG,
        "AxisNumber3": ULONG,
    }
    sampleSurfaceNormalObj = create_static_pv("SampleSurfaceNormalDirection", sampleSurfaceNormalType, sample_surface_normal_direction_record)
    server.addRecord("SampleSurfaceNormalDirection", sampleSurfaceNormalObj)

    detectorSetupType = {
        "PixelDirection1": STRING,
        "PixelDirection2": STRING,
        "CenterChannelPixel": [ULONG],
        "Size": [DOUBLE],
        "Distance": DOUBLE,
        "Units": STRING,
    }
    detectorSetupObj = create_static_pv("DetectorSetup", detectorSetupType, detector_setup_record)
    server.addRecord("DetectorSetup", detectorSetupObj)

    # Display available channel names.
    print("CHANNELS: %s" % server.getRecordNames())

    # -------------------------------
    # Dynamic Update Loop for Axis Records
    # -------------------------------
    base_positions = {name: params["Position"] for name, params in axis_records.items()}
    amplitude = 0.5        # amplitude of sine-wave offset
    update_interval = 0.5  # seconds between updates
    startTime = time.time()

    try:
        while True:
            elapsed = time.time() - startTime
            # Update each axis record's "Position" field.
            for pvname, base in base_positions.items():
                new_position = base + amplitude * math.sin(elapsed)
                newPv = create_axis_pv(pvname, {
                    "AxisNumber": axis_records[pvname]["AxisNumber"],
                    "SpecMotorName": axis_records[pvname]["SpecMotorName"],
                    "DirectionAxis": axis_records[pvname]["DirectionAxis"],
                    "Position": new_position,
                })
                server.update(pvname, newPv)
            time.sleep(update_interval)
    except KeyboardInterrupt:
        print("\nShutting down the PvaServer.")
