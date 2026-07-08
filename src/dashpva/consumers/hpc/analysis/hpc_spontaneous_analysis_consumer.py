# Copyright © 2026, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: DashPVA
# By: Argonne National Laboratory
#
# BSD OPEN SOURCE LICENSE
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# ******************************************************************************************************
# DISCLAIMER
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ******************************************************************************************************

import time

import numpy as np
import pvaccess as pva
import toml
from pvaccess import DOUBLE, PvObject
from pvapy.utility.timeUtility import TimeUtility

from dashpva.consumers.core.base_analysis_processor import BaseAnalysisProcessor


# Example AD Metadata Processor for the streaming framework
# This updated version processes one frame at a time.
class HpcAnalysisProcessor(BaseAnalysisProcessor):

    def __init__(self, configDict={}):
        super().__init__(configDict)
        # Configuration parameters
        self.configure(configDict)

    def configure(self, configDict):
        """
        Configure user-defined settings from configDict if needed.
        """
        if 'path' in configDict:
            self.path = configDict['path']
        else:
            import dashpva.settings as _settings
            self.path = _settings.ensure_path()
            if self.path is None:
                raise RuntimeError(
                    "HpcAnalysisProcessor: no 'path' in configDict and "
                    "no effective config path is available — configure a profile first."
                )

        with open(self.path, 'r') as f:
            self.config: dict = toml.load(f)

        self.axis1 = self.config.get('ANALYSIS', {}).get('AXIS1', None)
        self.axis2 = self.config.get('ANALYSIS', {}).get('AXIS2', None)

    def pva_to_image(self, pva_object):
        """
        Convert the PVA Object to a NumPy array representing the image.
        Apply correct shaping and transpose if needed.
        """
        try:

            self.image = None
            if pva_object is not None and 'dimension' in pva_object:
                dims = pva_object['dimension']
                shape = tuple([dim['size'] for dim in dims])
                raw_data = np.array(pva_object['value'][0][self.data_type])
                # Reshape and transpose if necessary to get correct orientation
                self.image = np.reshape(raw_data, shape).T
        except Exception:
            print("error parsing images")


    def process(self, pvObject):
        """
        Process each incoming frame individually.
        Steps:
          1. Parse attributes and image data.
          2. Compute ROI-based intensity and center-of-mass (COM).
          3. Get current frame's X, Y from attributes.
          4. Append these analysis results as an NtAttribute to the pvObject.
        """
        t0 = time.time()

        # Retrieve frame id
        pvObject['uniqueId']
        dims = pvObject['dimension']
        nDims = len(dims)
        if not nDims:
            # Frame has no image data
            return pvObject

        if 'timeStamp' not in pvObject:
            # No timestamp, just return the object
            return pvObject

        # Parse attributes and image type
        self.parse_pva_ndattributes(pvObject)
        self.parse_image_data_type(pvObject)
        self.pva_to_image(pvObject)

        if self.image is None:
            # If we cannot form the image, skip analysis
            return pvObject

        # Extract X, Y positions from attributes as they come in
        # The original code accessed x,y as: attributes.get('x')[0]['value']
        # Adjust as needed depending on attribute structure.
        if self.axis1 is not None and self.axis2 is not None:
            x_attr = self.attributes.get(self.axis1, None)
            y_attr = self.attributes.get(self.axis2, None)
            if x_attr is not None and y_attr is not None:
                x_value = x_attr[0]['value'] if isinstance(x_attr, tuple) else 0.0
                y_value = y_attr[0]['value'] if isinstance(y_attr, tuple) else 0.0
        else:
            # Default to 0 if attributes not found
            x_value = 0.0
            y_value = 0.0

        # Extract Region of Interest (ROI) from the image
        roi = self.image[self.roi_y:self.roi_y+self.roi_height,
                         self.roi_x:self.roi_x+self.roi_width]

        # Compute intensity (sum of ROI pixels)
        intensity = np.sum(roi)

        # Compute center-of-mass (COM)
        # To avoid division by zero, check intensity
        if intensity <= 0:
            com_x = 0.0
            com_y = 0.0
        else:
            y_coords, x_coords = np.indices(roi.shape)
            weighted_sum_x = np.sum(roi * x_coords)
            weighted_sum_y = np.sum(roi * y_coords)
            com_x = weighted_sum_x / intensity
            com_y = weighted_sum_y / intensity

        # Now create a PvObject with the analysis results
        # We will send out a single data point (X, Y, Intensity, ComX, ComY)
        analysis_object = PvObject({'value':{'Axis1': DOUBLE, 'Axis2': DOUBLE,
                                             'Intensity': DOUBLE,
                                             'ComX': DOUBLE,
                                             'ComY': DOUBLE}},
                                   {'value':{'Axis1': float(x_value),'Axis2': float(y_value),
                                             'Intensity': float(intensity),
                                             'ComX': float(com_x),
                                             'ComY': float(com_y)}})

        # Create an NtAttribute to hold this analysis data
        pvAttr = pva.NtAttribute('Analysis', analysis_object)

        # Append this attribute to the frame's attribute list
        frameAttributes = pvObject['attribute']
        frameAttributes.append(pvAttr)
        pvObject['attribute'] = frameAttributes

        # Update stats
        frameTimestamp = TimeUtility.getTimeStampAsFloat(pvObject['timeStamp'])
        self.lastFrameTimestamp = frameTimestamp
        self.nFramesProcessed += 1

        # Update output channel if needed
        self.updateOutputChannel(pvObject)

        # Update processing time
        t1 = time.time()
        self.processingTime += (t1 - t0)

        return pvObject
