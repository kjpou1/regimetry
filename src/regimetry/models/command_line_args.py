from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CommandLineArgs:
    command: str
    config: str
    debug: bool
    signal_data_dir: Optional[str] = None
