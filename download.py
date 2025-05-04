import argparse
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from huggingface_hub.hf_api import HfApi
from modelscope import HubApi
from modelscope.hub.file_download import model_file_download
from modelscope.hub.snapshot_download import snapshot_download

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
parser.add_argument(
    "--modelscope",
    action="store_true",
    help="Use this flag if the model is from ModelScope.",
)

args = parser.parse_args()

api_hf = HfApi(token=os.getenv("HF_TOKEN"))
api_ms = HubApi()

model_id = args.input
output_dir = Path("models") / (
    args.output if args.output else model_id.replace("/", "_")
)
output_dir.mkdir(parents=True, exist_ok=True)

if args.modelscope:
    api_ms.login(os.getenv("MODELSCOPE_TOKEN"))

if args.file:
    print(f"Downloading {args.file} from {model_id} to {output_dir}")
    if args.modelscope:
        model_file_download(
            model_id=model_id,
            file_path=args.file,
            local_dir=output_dir,
        )
    else:
        api_hf.hf_hub_download(
            repo_id=model_id, filename=args.file, local_dir=output_dir
        )

else:
    print(f"Downloading {model_id} to {output_dir}")
    if args.modelscope:
        snapshot_download(
            model_id=model_id,
            local_dir=output_dir,
            ignore_file_pattern=args.exclude.split(",") if args.exclude else None,
        )
    else:
        api_hf.snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=args.exclude.split(",") if args.exclude else None,
        )

print("Done!")
