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

import pyFAI
import fabio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Paths to your files
poni_file = "2022-3_calib.poni"   # PONI file
mask_file = "2022-3_mask.edf"   # Mask file
image_file = "d350_CeO2-000000.tif"  # Data file (TIFF)
metadata_file = "d350_CeO2-000000.tif.metadata"  # Metadata file

# Load the image data (diffraction image)
try:
    data = fabio.open(image_file).data
except Exception as e:
    print(f"Error loading image file: {e}")
    data = None

# Load the mask
try:
    mask = fabio.open(mask_file).data
except Exception as e:
    print(f"Error loading mask file: {e}")
    mask = None

# Load the PONI file for geometry calibration
try:
    ai = pyFAI.load(poni_file)
except Exception as e:
    print(f"Error loading PONI file: {e}")
    ai = None

# Read the metadata from the tiff.metadata file
metadata = ""
try:
    with open(metadata_file, 'r') as file:
        metadata = file.read()
except FileNotFoundError:
    metadata = "Metadata file not found."
except Exception as e:
    metadata = f"Error reading metadata file: {e}"

# Check if all necessary files were loaded
if data is not None and mask is not None and ai is not None:
    # Perform azimuthal integration
    q, intensity = ai.integrate1d(data, 1000, mask=mask, unit="q_A^-1", method='bbox')

    # Plot the diffraction image (TIFF)
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap='gray', origin='lower', norm =LogNorm())
    plt.colorbar(label='Intensity')
    plt.title("Diffraction Image")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap='gray', origin='lower', norm =LogNorm())
    plt.colorbar(label='Intensity')
    plt.title("Mask")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.show()

    # Plot the azimuthal integration result
    plt.figure(figsize=(8, 6))
    plt.plot(q, intensity, label="Diffraction Pattern")
    plt.xlabel(r"Q (Å$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")
    plt.title("Azimuthal Integration of Diffraction Data for CeO2")
    plt.legend()
    plt.grid(True)
    plt.show()

else:
    print("One or more files could not be loaded. Please check the file paths and try again.")

# Print out the metadata
print("\nMetadata:")
print(metadata)



# import pyFAI
# import fabio
# import numpy as np
# import matplotlib.pyplot as plt

# # Paths to your files
# data_file = "2022-3_calib.poni"   # Replace with your PONI file path
# mask_file = "2022-3_mask.edf"   # Replace with your mask file path
# image_file = "d350_CeO2-000000.tif"  # Replace with your data file path

# # Load the image data (diffraction image)
# data = fabio.open(image_file).data

# # Load the mask
# mask = fabio.open(mask_file).data

# # Load the PONI file for geometry calibration
# ai = pyFAI.load("2022-3_calib.poni")

# # Perform azimuthal integration
# q, intensity = ai.integrate1d(data, 1000, mask=mask, unit="q_A^-1", method='bbox')

# # Plot the results
# plt.figure(figsize=(8, 6))
# plt.plot(q, intensity, label="Diffraction Pattern")
# plt.xlabel(r"Q (Å$^{-1}$)")
# plt.ylabel("Intensity (a.u.)")
# plt.title("Azimuthal Integration of Diffraction Data for CeO2")
# plt.legend()
# plt.grid(True)
# plt.show()
