"""Compile a Cell Ontology OBO file into deterministic PyTorch tensors used by scOLAR."""

import argparse
import os
import re
import unicodedata
from collections import deque

import torch
import obonet
import igraph as ig
from tqdm import tqdm


def norm(s: str) -> str:
    """Normalize text with Unicode NFKC, whitespace trimming, and case folding."""
    if s is None:
        return ""
    return unicodedata.normalize("NFKC", str(s)).strip().casefold()


_CL_ID_RE = re.compile(r"(CL:\d{7})")


def _clean_cl_id(raw: str) -> str:
    """Extract a canonical CL identifier from an annotated OBO relationship value."""
    m = _CL_ID_RE.search(raw)
    return m.group(1) if m else ""


class OntologyCompiler:
    """Parse Cell Ontology and build the mappings and graph tensors required for training."""

    def __init__(self, ontology_path: str):
        print(f"[Compiler] Loading ontology from {ontology_path}...")
        self.obonet_graph = obonet.read_obo(ontology_path)
        self.graph = ig.Graph(directed=True)
        self.terms = []
        self.id2idx = {}
        self.term_map = {}

    def build_graph(self):
        print("[Compiler] Parsing terms and building DAG...")

        valid_terms = []
        for term_id in tqdm(self.obonet_graph, desc="Scanning terms"):
            if not term_id.startswith("CL:"):
                continue
            term_data = self.obonet_graph.nodes[term_id]
            if term_data.get("is_obsolete", "false") == "true":
                continue
            valid_terms.append(term_id)

        valid_terms.sort()

        for idx, term_id in enumerate(valid_terms):
            self.terms.append(term_id)
            self.id2idx[term_id] = idx
            self.graph.add_vertex(name=term_id)

            term_data = self.obonet_graph.nodes[term_id]

            self.term_map[norm(term_id)] = idx

            name = term_data.get("name")
            if name:
                self.term_map[norm(name)] = idx

            for syn in term_data.get("synonym", []):
                syn_upper = syn.upper()
                if any(t in syn_upper for t in ("EXACT", "RELATED", "BROAD")):
                    if '"' in syn:
                        desc = syn.split('"')[1].strip()
                        if desc:
                            self.term_map[norm(desc)] = idx

        print(f"[Compiler] Registered {len(self.terms)} valid CL terms.")
        print(
            f"[Compiler] Registered {len(self.term_map)} semantic keys " f"(ID + name + synonyms)."
        )

        edge_list = []
        skipped = 0
        for term_id in tqdm(valid_terms, desc="Building edges"):
            child_idx = self.id2idx[term_id]
            term_data = self.obonet_graph.nodes[term_id]

            for raw_parent in term_data.get("is_a", []):
                parent_id = _clean_cl_id(raw_parent)
                if parent_id and parent_id in self.id2idx:
                    edge_list.append((child_idx, self.id2idx[parent_id]))
                else:
                    skipped += 1

        self.graph.add_edges(edge_list)
        print(
            f"[Compiler] Graph: {len(edge_list)} is_a edges added, "
            f"{skipped} skipped (non-CL parents, expected for cross-ontology refs)."
        )

        if len(edge_list) == 0:
            raise RuntimeError(
                "[Compiler] FATAL: 0 edges added. "
                "Check obonet version and OBO file format. "
                "is_a values may not be parsed correctly."
            )

    def compile_matrices(self):
        num_nodes = len(self.terms)
        print(f"[Compiler] Compiling matrices for {num_nodes} nodes...")

        parent_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        for e in self.graph.es():
            parent_matrix[e.source, e.target] = True

        ancestor_incl = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        ancestor_excl = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)

        for i in tqdm(range(num_nodes), desc="Computing ancestors"):
            ancestors = self.graph.subcomponent(i, mode="out")
            for j in ancestors:
                ancestor_incl[i, j] = True
                if j != i:
                    ancestor_excl[i, j] = True

        depths = self._compute_depths(num_nodes)

        return parent_matrix, ancestor_incl, ancestor_excl, depths

    def _compute_depths(self, num_nodes: int) -> torch.Tensor:
        """Compute the shortest directed distance from every term to the Cell Ontology root."""
        depths = torch.full((num_nodes,), fill_value=-1, dtype=torch.long)

        root_id = "CL:0000000"
        root_idx = self.id2idx.get(root_id, -1)

        if root_idx == -1:
            print(f"[Compiler] WARNING: Root ({root_id}) not found. " f"All depths set to -1.")
            return depths

        try:

            # distances[i] = [distance from vertex i to root_idx]
            raw_distances = self.graph.distances(target=root_idx, weights=None, mode="out")

            dist_list = [row[0] for row in raw_distances]

            if len(dist_list) != num_nodes:
                raise ValueError(
                    f"Expected {num_nodes} distance values, "
                    f"got {len(dist_list)}. Falling back to BFS."
                )

            safe_dist = [int(d) if (d != float("inf") and d == d) else -1 for d in dist_list]
            depths = torch.tensor(safe_dist, dtype=torch.long)

            n_unreachable = (depths == -1).sum().item()
            if n_unreachable > 0:
                print(
                    f"[Compiler] {n_unreachable} nodes unreachable from root "
                    f"(disconnected subgraph). depth=-1 for those nodes."
                )

        except Exception as e:
            print(f"[Compiler] igraph distances failed ({e}). " f"Using BFS fallback...")
            depths = self._bfs_depths(num_nodes, root_idx)

        if depths[root_idx].item() != 0:
            print(
                f"[Compiler] WARNING: Root node depth = {depths[root_idx].item()}, "
                f"expected 0. Check graph direction."
            )

        print(
            f"[Compiler] Depth stats: "
            f"min={depths[depths >= 0].min().item()}, "
            f"max={depths.max().item()}, "
            f"mean={depths[depths >= 0].float().mean().item():.2f}"
        )

        return depths

    def _bfs_depths(self, num_nodes: int, root_idx: int) -> torch.Tensor:
        """Compute root distances with a breadth-first fallback over child-to-parent edges."""
        depths = torch.full((num_nodes,), fill_value=-1, dtype=torch.long)
        depths[root_idx] = 0

        queue = deque([root_idx])
        visited = {root_idx}

        while queue:
            curr = queue.popleft()
            curr_depth = depths[curr].item()

            for child in self.graph.neighbors(curr, mode="in"):
                if child not in visited:
                    visited.add(child)
                    depths[child] = curr_depth + 1
                    queue.append(child)

        return depths

    def save(self, output_path: str):
        if not self.terms:
            raise ValueError("Graph not built. Call build_graph() first.")

        parent_matrix, ancestor_incl, ancestor_excl, depths = self.compile_matrices()

        n = len(self.terms)
        n_anc_links = ancestor_excl.sum().item()
        print(f"\n[Compiler] Sanity check:")
        print(f"  Nodes           : {n}")
        print(f"  is_a edges      : {self.graph.ecount()}")
        print(f"  Ancestor links  : {n_anc_links}  " f"(ancestor_excl_self, expected >> 0)")
        print(f"  term_map keys   : {len(self.term_map)}")
        if n_anc_links == 0:
            raise RuntimeError(
                "[Compiler] FATAL: ancestor_excl is all-zero. "
                "is_a edges were not added correctly. "
                "This would silently break HCL and Adversarial losses."
            )

        data = {
            "meta": {
                "description": "Compiled Cell Ontology for scOLAR",
                "version": self.obonet_graph.graph.get("data-version", "unknown"),
                "depth_definition": "distance_to_root_upward",
                "n_terms": n,
                "n_edges": self.graph.ecount(),
            },
            "mappings": {
                "id2idx": self.id2idx,
                "idx2id": {v: k for k, v in self.id2idx.items()},
                "term_map": self.term_map,
            },
            "structure": {
                "parent_matrix": parent_matrix,  # [N,N] bool
                "ancestor_matrix_incl_self": ancestor_incl,  # [N,N] bool
                "ancestor_matrix_excl_self": ancestor_excl,  # [N,N] bool
                "depths": depths,  # [N]   long
                "all_terms": self.terms,  # list[str]
            },
        }

        print(f"\n[Compiler] Saving to {output_path}...")
        torch.save(data, output_path)

        file_mb = os.path.getsize(output_path) / (1024**2)
        print(f"[Compiler] Done. File size: {file_mb:.2f} MB")
        print(f"[Compiler] Load with: OntologyManager('{output_path}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile Cell Ontology (.obo) to PyTorch tensors for scOLAR"
    )
    parser.add_argument("--input", type=str, required=True, help="Path to cl.obo")
    parser.add_argument(
        "--output",
        type=str,
        default="data/cl.pt",
        help="Output path for compiled .pt file (default: data/cl.pt)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[Error] Input file not found: {args.input}")
        exit(1)

    compiler = OntologyCompiler(args.input)
    compiler.build_graph()
    compiler.save(args.output)
