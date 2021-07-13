import argparse
import numpy
import torch
import pickle
import shutil

from pathlib import Path


def convert_dglke_embeddings(emb_file: Path, data_dir: Path, output_dir: Path):
    print(f"Loading embeddings from {emb_file}")
    npy_data = numpy.load(str(emb_file))
    torch_data = torch.tensor(npy_data)
    print(f"Found: {torch_data.shape}")

    entity_emb_file = output_dir / "entity_embeddings.pkl"
    print(f"Save embeddings to {output_dir}")
    pickle.dump(torch_data, entity_emb_file.open("wb"))

    shutil.copyfile(data_dir / "entities.dict", output_dir / "entities.dict")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_file", type=Path, required=True,
                        help="Path to the entity embedding file")
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Path to the directory containing the training files")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Path to the output directory")
    args = parser.parse_args()

    convert_dglke_embeddings(args.emb_file, args.data_dir, args.output_dir)
