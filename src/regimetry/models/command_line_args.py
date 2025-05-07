from dataclasses import dataclass
from typing import Optional

@dataclass
class CommandLineArgs:
    command: str
    config: Optional[str]
    debug: bool
    signal_input_path: Optional[str] = None
