#!/usr/bin/env python
"""
This script creates a PvaServer with multiple channels.
Each axis channel is updated periodically (its "Position" field oscillates),
while the UBMatrix and Energy channels remain static.
"""

import time
import math
from pvaccess import ULONG, STRING, DOUBLE
from pvaccess import PvObject, PvaServer

# -------------------------------
# Define PV record parameters
# -------------------------------

# Axis PV definitions: each record holds an axis number, a Spec motor name,
# a direction string, and a position value that will be updated.
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

# Static PVs: UBMatrix and Energy records
ub_matrix_record = {
    "Value": [1, 0, 0, 0, 1, 0, 0, 0, 1]
}

energy_record = {
    "Value": 11.212  # keV
}

# -------------------------------
# Helper functions to create PvObjects
# -------------------------------

def create_axis_pv(initial):
    """
    Creates a PvObject for an axis record.
    The type dictionary specifies:
      - "AxisNumber": ULONG
      - "SpecMotorName": STRING
      - "DirectionAxis": STRING
      - "Position": DOUBLE
    """
    typeDict = {
        "AxisNumber": ULONG,
        "SpecMotorName": STRING,
        "DirectionAxis": STRING,
        "Position": DOUBLE,
    }
    return PvObject(typeDict, initial)

def create_static_pv(typeDict, initial):
    """
    Creates a PvObject for a static record.
    """
    return PvObject(typeDict, initial)

# -------------------------------
# Main server setup
# -------------------------------

if __name__ == '__main__':
    # Create a single PvaServer instance.
    server = PvaServer()

    # Dictionary to store the current PvObject for each axis record.
    axis_pv_objects = {}

    # Add each axis record.
    for pvname, params in axis_records.items():
        pvObj = create_axis_pv(params)
        server.addRecord(pvname, pvObj)
        axis_pv_objects[pvname] = pvObj

    # Add the static UBMatrix record.
    # For an array we let the PvObject infer type from the Python list.
    ubMatrixType = {"Value": [ULONG]}
    ubMatrixObj = create_static_pv(ubMatrixType, ub_matrix_record)
    server.addRecord("6idb:spec:UB_matrix", ubMatrixObj)

    # Add the static Energy record.
    energyType = {"Value": DOUBLE}
    energyObj = create_static_pv(energyType, energy_record)
    server.addRecord("6idb:spec:Energy", energyObj)

    # Display available channel names.
    print("CHANNELS: %s" % server.getRecordNames())

    # For each axis record, store its base (initial) Position.
    base_positions = {name: params["Position"] for name, params in axis_records.items()}

    # Settings for periodic update of axis "Position"
    amplitude = 0.5        # amplitude of sine-wave offset
    update_interval = 0.5  # seconds between updates
    startTime = time.time()

    try:
        while True:
            elapsed = time.time() - startTime
            # Update each axis record's "Position" field.
            for pvname, base in base_positions.items():
                # Calculate a new position: base + amplitude * sin(elapsed)
                new_position = base + amplitude * math.sin(elapsed)
                # Prepare a new PvObject with the updated value.
                newPv = create_axis_pv({
                    "AxisNumber": axis_records[pvname]["AxisNumber"],
                    "SpecMotorName": axis_records[pvname]["SpecMotorName"],
                    "DirectionAxis": axis_records[pvname]["DirectionAxis"],
                    "Position": new_position,
                })
                # Update the record on the server.
                server.update(pvname, newPv)
            time.sleep(update_interval)
    except KeyboardInterrupt:
        print("\nShutting down the PvaServer.")
