#!/usr/bin/env python3
"""
This script sets up a CA IOC (using pva.CaIoc) to broadcast a set of PVs
exclusively via the Channel Access (CA) protocol. It includes:

  - Dynamic axis records (6idb1:m28.RBV, etc.) whose "Position" field is updated periodically.
  - Static records:
      * 6idb:spec:UB_matrix
      * 6idb:spec:Energy
      * PrimaryBeamDirection
      * InplaneReferenceDirection
      * SampleSurfaceNormalDirection
      * DetectorSetup

Instead of sending dictionary data as JSON strings, we create individual records
for each field in the dictionaries.
"""

import os
import time
import tempfile
import ctypes.util
import math
import numpy as np
import pvaccess as pva  # pva module provides CaIoc and related functions
from epics import camonitor, caget

# -------------------------------
# Define PV Record Parameters
# -------------------------------

# Dynamic axis records:
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
        "Position": 0.0,
    },
    "6idb1:m19.RBV": {
        "AxisNumber": 3,
        "SpecMotorName": "Chi",
        "DirectionAxis": "y+",
        "Position": 0.0
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
        "Position": 0.0,
    },
}

for key in axis_records.keys():
    axis_records[key]["Name"] = key

# Static records:

# Existing static records:
ub_matrix_record = {
    "Value": [1, 0, 0, 0, 1, 0, 0, 0, 1]
}
energy_record = {
    "Value": 5.212  # keV
}

# Additional static records based on HKL configuration:
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
    "CenterChannelPixel": [500, 500],
    "Size": [100, 100],
    "Distance": 400.644,
    "Units": "mm"
}

# -------------------------------
# Helper: Convert to Valid EPICS Record Name
# -------------------------------

def valid_record_name(name) -> str:
    """
    Converts the given record name to a valid EPICS record name.
    In this example, any '.' characters are replaced with '_'.
    """
    return name.replace('.', '_')

# -------------------------------
# CA IOC Setup Functions
# -------------------------------

def get_record_definition(name, value_type) -> str:
    """
    Returns the appropriate EPICS record definition based on the value type.
    """
    if isinstance(value_type, (int, float)):
        return """
        record(ai, "%s") {
            field(DTYP, "Soft Channel")
            field(PREC, "6")
        }
        """ % name
    elif isinstance(value_type, str):
        return """
        record(stringout, "%s") {
            field(DTYP, "Soft Channel")
            field(VAL, "")
        }
        """ % name
    elif isinstance(value_type, list) and all(isinstance(x, (int, float)) for x in value_type):
        # For numeric arrays (like UB_matrix)
        return """
        record(waveform, "%s") {
            field(DTYP, "Soft Channel")
            field(FTVL, "DOUBLE")
            field(NELM, "9")
        }
        """ % name

def setup_ca_ioc(records_dict) -> pva.CaIoc:
    """
    Sets up a CA IOC using pva.CaIoc to broadcast records with the given names and value types.
    """
    if not os.environ.get('EPICS_DB_INCLUDE_PATH'):
        pvDataLib = ctypes.util.find_library('pvData')
        if not pvDataLib:
            raise Exception('Cannot find dbd directory. Please set EPICS_DB_INCLUDE_PATH.')
        pvDataLib = os.path.realpath(pvDataLib)
        epicsLibDir = os.path.dirname(pvDataLib)
        dbdDir = os.path.realpath('%s/../../dbd' % epicsLibDir)
        os.environ['EPICS_DB_INCLUDE_PATH'] = dbdDir
        
    # Create a temporary database file
    dbFile = tempfile.NamedTemporaryFile(delete=False, mode='w')
    
    # Create all the record definitions
    for base_name, record_data in records_dict.items():
        valid_base = valid_record_name(base_name)
        
        # Add the Name field as a normal field
        record_data = dict(record_data)  # Make a copy
        record_data["Name"] = base_name  # Add original name
        
        # Create individual records for each field in the record data
        for field_name, field_value in record_data.items():
            record_name = "%s:%s" % (valid_base, field_name)
            dbFile.write(get_record_definition(record_name, field_value))

    dbFile.close()

    caIoc = pva.CaIoc()
    caIoc.loadDatabase('base.dbd', '', '')
    caIoc.registerRecordDeviceDriver()
    caIoc.loadRecords(dbFile.name, '')
    print(caIoc.getRecordNames())
    caIoc.start()
    os.unlink(dbFile.name)
    
    return caIoc

def update_ca_record_field(caIoc, base_name, field_name, value) -> None:
    """
    Updates a specific field of a CA record.
    """
    valid_base = valid_record_name(base_name)
    record_name = "%s:%s" % (valid_base, field_name)
    
    try:
        # print(record_name, value)
        if isinstance(value, (list, np.ndarray)) and all(isinstance(x, (int, float, np.float32)) for x in value):
            # For numeric arrays
            value = [int(val) for val in value]
            caIoc.putField(record_name, value)
        else:
        #     # For scalar values or other types
            caIoc.putField(record_name, value)
    except Exception as e:
        print("Error updating %s: %s" % (record_name, e))

def update_full_record(caIoc, base_name, record_data):
    """
    Updates all fields of a record based on the dictionary data.
    """
    for field_name, field_value in record_data.items():
        update_ca_record_field(caIoc, base_name, field_name, field_value)

# -------------------------------
# Main Loop for Updates
# -------------------------------

def main() -> None:
    # Combine all records into one dictionary
    all_records = {
        **axis_records,
        "6idb:spec:UB_matrix": ub_matrix_record,
        "6idb:spec:Energy": energy_record,
        "PrimaryBeamDirection": primary_beam_direction_record,
        "InplaneReferenceDirection": inplane_reference_direction_record,
        "SampleSurfaceNormalDirection": sample_surface_normal_direction_record,
        "DetectorSetup": detector_setup_record
    }
    print(axis_records)
    
    # Set up the CA IOC with these records
    caIoc = setup_ca_ioc(all_records)

    # Add Name field to all static records
    static_records = {
        "6idb:spec:UB_matrix": {**ub_matrix_record, "Name": "6idb:spec:UB_matrix"},
        "6idb:spec:Energy": {**energy_record, "Name": "6idb:spec:Energy"},
        "PrimaryBeamDirection": {**primary_beam_direction_record, "Name": "PrimaryBeamDirection"},
        "InplaneReferenceDirection": {**inplane_reference_direction_record, "Name": "InplaneReferenceDirection"},
        "SampleSurfaceNormalDirection": {**sample_surface_normal_direction_record, "Name": "SampleSurfaceNormalDirection"},
        "DetectorSetup": {**detector_setup_record, "Name": "DetectorSetup"}
    }
    
    # Update static records (they remain constant)
    for rec_name, rec_data in all_records.items():
        update_full_record(caIoc, rec_name, rec_data)

    # For dynamic axis records, store their base positions
    base_positions = {name: rec['Position'] for name, rec in axis_records.items()}
    dynamic_records = {**axis_records, '6idb:spec:Energy':13.0,} #'6idb:spec:UB_matrix': caget('6idb:spec:UB_matrix')}

    # amplitude = 0.5        # Amplitude of the sine-wave update
    # update_interval = 0.5  # Seconds between updates
    # start_time = time.time()

    try:
        while True:
            # elapsed = time.time() - start_time
            for name, rec in dynamic_records.items():
                # Update the Position field with a sine offset
                new_position = caget(name) #+ amplitude * math.sin(elapsed)
                # Only update the Eta field For Now
                if isinstance(rec, dict):
                    # if rec["SpecMotorName"] == "Eta":
                    #     # Only Update Eta Position
                    update_ca_record_field(caIoc, name, 'Position', new_position)
                elif name == '6idb:spec:Energy':
                    update_ca_record_field(caIoc, name, 'Value', new_position)
                # elif name == '6idb:spec:UB_matrix':
                #     update_ca_record_field(caIoc, name, 'Value', new_position)
            
            # time.sleep(update_interval)
    except KeyboardInterrupt:
        print("Shutting down CA IOC updates.")

if __name__ == '__main__':
    main()