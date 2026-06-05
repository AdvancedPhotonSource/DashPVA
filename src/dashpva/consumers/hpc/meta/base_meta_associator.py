import time

import numpy as np
import pvaccess as pva
from pvapy.utility.timeUtility import TimeUtility

from dashpva.consumers.hpc.base_hpc import BaseHpcProcessor


class BaseMetaAssociator(BaseHpcProcessor):

    # Static geometry PVs never update, so their timestamps are hours old by
    # design. A tight default discarded every one of them and starved the
    # analysis consumers of HKL attributes, so the gate is off unless a config
    # asks for it.
    DEFAULT_TIMESTAMP_TOLERANCE = float('inf')
    DEFAULT_METADATA_TIMESTAMP_OFFSET = 0.001

    def __init__(self, configDict={}):
        super().__init__(configDict)
        self.timestampTolerance = float(configDict.get('timestampTolerance', self.DEFAULT_TIMESTAMP_TOLERANCE))
        self.metadataTimestampOffset = float(configDict.get('metadataTimestampOffset', self.DEFAULT_METADATA_TIMESTAMP_OFFSET))
        self.processor_id = configDict.get('collectorId') if 'collectorId' in configDict else configDict.get('metadataId', None)
        self.currentMetadataMap = {}
        self.cd = None

    def configure(self, configDict):
        self.cd = configDict
        if 'timestampTolerance' in configDict:
            self.timestampTolerance = float(configDict['timestampTolerance'])
            self.logger.debug(f'Updated timestamp tolerance: {self.timestampTolerance} seconds')
        if 'metadataTimestampOffset' in configDict:
            self.metadataTimestampOffset = float(configDict['metadataTimestampOffset'])
            self.logger.debug(f'Updated metadata timestamp offset: {self.metadataTimestampOffset} seconds')

    def associateMetadata(self, mdChannel, frameId, frameTimestamp, frameAttributes):
        if mdChannel not in self.currentMetadataMap:
            # Metadata for this channel has not arrived, so it cannot be
            # attached. A routine discard, not a fault: counted for the status
            # channel, not logged -- warning per channel per frame buried real
            # errors under thousands of lines.
            self.nMetadataDiscarded += 1
            return False

        mdObject = self.currentMetadataMap[mdChannel]

        if 'timeStamp' in mdObject:
            mdTimestamp = TimeUtility.getTimeStampAsFloat(mdObject['timeStamp'])
            mdTimestamp2 = mdTimestamp + self.metadataTimestampOffset

        if 'value' not in mdObject:
            self.logger.error(f'Metadata object {mdObject} does not have field "value"')
            return False

        mdValue = mdObject['value']
        self.logger.debug(f"Value from metadata object: {mdValue}")
        try:
            if isinstance(mdValue, (int, float)):
                mdValue = float(mdValue)
                nt_attribute = {'name': mdChannel, 'value': pva.PvFloat(mdValue)}
            elif isinstance(mdValue, str):
                nt_attribute = {'name': mdChannel, 'value': pva.PvString(mdValue)}
            elif isinstance(mdValue, np.ndarray):
                pv = pva.PvScalarArray(pva.DOUBLE)
                pv.set(mdValue.tolist())
                nt_attribute = {'name': mdChannel, 'value': pv}
            elif isinstance(mdValue, bool):
                nt_attribute = {'name': mdChannel, 'value': pva.PvBoolean(mdValue)}
            else:
                raise ValueError(f'Failed to create metadata attribute: {mdChannel}: {mdValue}')

            frameAttributes.append(nt_attribute)
        except Exception as e:
            self.logger.error(f"[Metadata Associator] Error associatating metadata {e}")
            return False

        diff = abs(frameTimestamp - mdTimestamp2)
        self.logger.debug(f'Metadata {mdChannel} has value of {mdValue}, timestamp: {mdTimestamp} (with offset: {mdTimestamp2}), timestamp diff: {diff}')
        if diff > self.timestampTolerance:
            now = time.time()
            if now - self._lastToleranceWarnTime >= self._toleranceWarnIntervalSec:
                suppressed = self._toleranceWarnSuppressed
                suffix = f' ({suppressed} similar warnings suppressed)' if suppressed else ''
                self.logger.warning(
                    f'[Metadata Associator] Rejecting {mdChannel}: timestamp diff {diff:.6f}s exceeds tolerance {self.timestampTolerance}s{suffix}'
                )
                self._lastToleranceWarnTime = now
                self._toleranceWarnSuppressed = 0
            else:
                self._toleranceWarnSuppressed += 1
            self.nMetadataDiscarded += 1
            return False
        self.nMetadataProcessed += 1
        return True
