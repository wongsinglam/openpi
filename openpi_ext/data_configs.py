"""Extension data-config definitions."""

import dataclasses

from openpi.training.config import LeRobotLiberoDataConfig


@dataclasses.dataclass(frozen=True)
class LeRobotDobotFullDataConfig(LeRobotLiberoDataConfig):
    """Dobot full-task config.

    This currently reuses the Libero data pipeline as a starting point.
    Customize transforms here when your Dobot dataset schema is finalized.
    """

