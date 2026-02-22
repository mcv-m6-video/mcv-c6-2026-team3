from dataclasses import dataclass
from pathlib import Path
from argparse import Namespace
import os

@dataclass
class Config:
    
    xml_path : Path
    input_path : Path
    output_path : Path
    

def build_config(args: Namespace, result_subfolder : str) -> Config:
    input_path = Path(f"{args.data}/AICity_data/train/S03/c010/vdo.avi")
    output_path = Path(f"{args.results}/{result_subfolder}")
    os.makedirs(output_path, exist_ok=True)
    xml_path = Path(f"{args.data}/ai_challenge_s03_c010-full_annotation.xml")
    
    return Config(
        xml_path=xml_path,
        input_path=input_path,
        output_path=output_path
    )