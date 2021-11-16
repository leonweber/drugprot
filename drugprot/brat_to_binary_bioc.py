from collections import Counter
from pprint import pprint
from typing import Tuple, List

import networkx as nx

from drugprot.parse_standoff import StandoffAnnotation

from argparse import ArgumentParser
from pathlib import Path

def get_qualified_paths(G: nx.DiGraph, source_type: str, target_type: str) -> List[Tuple[str]]:
    qualified_paths = []

    G_modified = nx.DiGraph()
    G_modified.add_nodes_from((i, d) for i, d in G.nodes.data() if not d.get("modifications"))

    for u, v, edge_data in G.edges.data():
        if u not in G_modified.nodes or v not in G_modified.nodes:
            continue
        if edge_data["type"] == "Cause":
            G_modified.add_edge(v, u, **edge_data)
        else:
            G_modified.add_edge(u, v, **edge_data)

    for source in [i for i, d in G_modified.nodes.data() if d["type"] == source_type]:
        for target in [i for i, d in G_modified.nodes.data() if d["type"] == target_type]:
            if source == target:
                continue
            for path in list(nx.all_simple_paths(G=G_modified, source=source, target=target)):
                qualified_path = [source]
                for i in range(len(path)-1):
                    u = path[i]
                    v = path[i+1]
                    edge_type = G_modified.edges[u, v]["type"]
                    event_type = G_modified.nodes[v]["type"]
                    if v != target:
                        qualified_path.extend((edge_type, event_type))
                    else:
                        qualified_path.append(edge_type)
                qualified_path.append(target)
                qualified_paths.append(tuple(qualified_path))

    return qualified_paths




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    all_qualified_paths = []
    path: Path
    for path in args.input:
        for file in path.glob("*txt"):
            with file.with_suffix(".txt").open() as f:
                txt_lines = f.readlines()
            with file.with_suffix(".a1").open() as f:
                a1_lines = f.readlines()
            with file.with_suffix(".a2").open() as f:
                a2_lines = f.readlines()

            ann = StandoffAnnotation(a1_lines=a1_lines, a2_lines=a2_lines)
            all_qualified_paths.extend([i[1:-1] for i in get_qualified_paths(G=ann.event_graph,
                                                  source_type="Gene_or_gene_product",
                                                  target_type="Gene_or_gene_product")])

    print(f"Found {len(all_qualified_paths)} examples")
    pprint(Counter(all_qualified_paths).most_common())




    with args.output.open("w") as f:
        pass  # write
