import time

import numpy as np
import pvaccess as pva
from pvapy.utility.timeUtility import TimeUtility

from dashpva.consumers.core.base_hpc import BaseHpcProcessor


class BaseMetaAssociator(BaseHpcProcessor):
    """Base for metadata-associating HPC consumers.

    Extends :class:`BaseHpcProcessor` with timestamp-tolerance handling and
    :meth:`associateMetadata`, which matches a metadata channel value against a
    frame's timestamp and appends it to the frame's NDAttributes. Subclasses
    implement ``process`` and drive the association per frame.

    Example:
        class MyAssociator(BaseMetaAssociator):
            def process(self, pvObject):
                ts = TimeUtility.getTimeStampAsFloat(pvObject['timeStamp'])
                attrs = pvObject['attribute']
                self.associateMetadata('my:pv', pvObject['uniqueId'], ts, attrs)
                pvObject['attribute'] = attrs
                self.updateOutputChannel(pvObject)
                return pvObject
    """

    DEFAULT_TIMESTAMP_TOLERANCE = 0.001
    DEFAULT_METADATA_TIMESTAMP_OFFSET = 0.001

    def __init__(self, configDict={}):
        super().__init__(configDict)
        self.timestampTolerance = float(configDict.get('timestampTolerance', self.DEFAULT_TIMESTAMP_TOLERANCE))
        self.metadataTimestampOffset = float(configDict.get('metadataTimestampOffset', self.DEFAULT_METADATA_TIMESTAMP_OFFSET))
        self.processor_id = configDict.get('collectorId') if 'collectorId' in configDict else configDict.get('metadataId', None)
        self.currentMetadataMap = {}
        self.cd = None
        self._lastToleranceWarnTime = 0.0
        self._toleranceWarnSuppressed = 0
        self._toleranceWarnIntervalSec = 60.0
        self._mdMissingChannels = {}
        self._lastMissingWarnTime = {}

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
            # attached to the frame. Tally per channel and warn at most once
            # per interval per channel (ERROR every frame floods the GUI box).
            count = self._mdMissingChannels.get(mdChannel, 0) + 1
            self._mdMissingChannels[mdChannel] = count
            now = time.time()
            if now - self._lastMissingWarnTime.get(mdChannel, 0.0) >= self._toleranceWarnIntervalSec:
                self.logger.warning(
                    f'[Metadata Associator] {mdChannel} not attaching: no metadata '
                    f'received yet (missed {count} times)'
                )
                self._lastMissingWarnTime[mdChannel] = now
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
