from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    new_lines = []
    with args.input.open() as f:
        lines = f.readlines()
        for line in tqdm(lines):
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) <= 1:
                continue

            mesh_id = "MESH:" + fields[1]
            gene_id = fields[4]
            actions = fields[9].split("|")
            for action in actions:
                new_lines.append("\t".join(["Chemical", mesh_id, "Gene", gene_id, action]))


    with args.output.open("w") as f:
        f.write("\n".join(new_lines))
