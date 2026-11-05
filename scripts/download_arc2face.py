"""Download the Arc2Face weights and the InsightFace antelopev2 models.

    python scripts/download_arc2face.py --root Arc2Face

creates

    Arc2Face/models/arc2face/        Arc2Face UNet
    Arc2Face/models/encoder/         Arc2Face text encoder
    Arc2Face/models/antelopev2/      InsightFace face detector + ArcFace

which are the defaults of ``scripts/sample_images.py`` (``--arc2face_models_dir Arc2Face/models
--insightface_root Arc2Face``). The weights are provided by the Arc2Face authors
(https://github.com/foivospar/Arc2Face) under their own terms - please read them before use.
The Stable Diffusion 1.5 base model is fetched automatically from the Hugging Face Hub on first use.
"""
import argparse
import os
import zipfile

import gdown
from huggingface_hub import hf_hub_download

ANTELOPEV2_GDRIVE_ID = "18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="Arc2Face", help="models are stored in <root>/models")
    args = p.parse_args()
    models = os.path.join(args.root, "models")
    os.makedirs(models, exist_ok=True)

    for filename in [
        "arc2face/config.json",
        "arc2face/diffusion_pytorch_model.safetensors",
        "encoder/config.json",
        "encoder/pytorch_model.bin",
    ]:
        hf_hub_download(repo_id="FoivosPar/Arc2Face", filename=filename, local_dir=models)

    antelope_dir = os.path.join(models, "antelopev2")
    os.makedirs(antelope_dir, exist_ok=True)
    zip_path = os.path.join(antelope_dir, "antelopev2.zip")
    gdown.download(id=ANTELOPEV2_GDRIVE_ID, output=zip_path, quiet=False)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(models)
    os.remove(zip_path)
    # Arc2Face uses its own ArcFace weights instead of the default glintr100 recognizer
    hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arcface.onnx", local_dir=antelope_dir)
    glint = os.path.join(antelope_dir, "glintr100.onnx")
    if os.path.exists(glint):
        os.remove(glint)
    print(f"Done. Models are in {models}")


if __name__ == "__main__":
    main()
