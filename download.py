import argparse
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from huggingface_hub.hf_api import HfApi

load_dotenv(find_dotenv(".env.local"))
load_dotenv(find_dotenv(".env"))

parser = argparse.ArgumentParser(
    description="Download and extract a file from huggingface."
)
parser.add_argument(
    "-i",
    "--input",
    type=str,
    required=True,
    help="Model ID to download from huggingface.",
)
parser.add_argument(
    "-f",
    "--file",
    type=str,
    default=None,
    help="Specific file to download instead of the entire repository.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default=None,
    help="Destination folder to save the model. Defaults to model ID.",
)
parser.add_argument(
    "-e",
    "--exclude",
    type=str,
    default=None,
    help="Comma-separated list of file extensions to exclude from the download.",
)

args = parser.parse_args()

api = HfApi(token=os.getenv("HF_TOKEN"))

model_id = args.input
output_dir = Path("models") / (
    args.output if args.output else model_id.replace("/", "_")
)
output_dir.mkdir(parents=True, exist_ok=True)

if args.file:
    print(f"Downloading {args.file} from {model_id} to {output_dir}")
    api.hf_hub_download(repo_id=model_id, filename=args.file, local_dir=output_dir)

else:
    print(f"Downloading {model_id} to {output_dir}")
    api.snapshot_download(
        repo_id=model_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=args.exclude.split(",") if args.exclude else None,
    )

print("Done!")
