# Copyright (C) UChicago Argonne, LLC
# See LICENSE file for details
def rotation_cycle(min,max):
            # generator for the rotation
            while True:
                for i in range(min,max):
                    yield i