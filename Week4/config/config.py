from dataclasses import dataclass
from pathlib import Path
from argparse import Namespace
import os

@dataclass
class Config:
    input_path : Path
    output_path : Path

def build_config(args: Namespace,  create : bool = True) -> Config:
    input_path = Path(f"{args.data}/vdo.avi")
    output_path = Path(f"{args.results}")
    if create:
        os.makedirs(output_path, exist_ok=True)
    
    return Config(
        input_path=input_path,
        output_path=output_path,
    )
