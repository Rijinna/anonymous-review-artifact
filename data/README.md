# Public data and ontology acquisition

Raw public datasets are intentionally not bundled. Download the following
processed AnnData files into this directory without changing the filenames:

| Dataset | Public download | Accession | Filename | Bytes |
|---|---|---|---|---:|
| Cao | http://cblast.gao-lab.org/Cao/Cao.h5ad | GSE98561 | `Cao.h5ad` | 141834660 |
| Quake 10x | http://cblast.gao-lab.org/Quake_10x/Quake_10x.h5ad | GSE109774 | `Quake_10x.h5ad` | 871364109 |
| Quake Smart-seq2 | http://cblast.gao-lab.org/Quake_Smart-seq2/Quake_Smart-seq2.h5ad | GSE109774 | `Quake_Smart-seq2.h5ad` | 1133523938 |
| Wagner | http://cblast.gao-lab.org/Wagner/Wagner.h5ad | GSE112294 | `Wagner.h5ad` | 726932869 |
| Zeisel 2018 | http://cblast.gao-lab.org/Zeisel_2018/Zeisel_2018.h5ad | SRP135960 | `Zeisel_2018.h5ad` | 2108117857 |

Each file must provide `obs["cell_ontology_class"]`. Count data are read from
`layers["counts"]` when present and otherwise from `X`.

An authoritative, complete set of SHA-256 values for all five original H5AD
inputs could not be recovered from the retained experiment records. No hashes
are inferred from filenames, download sizes, or later derived artifacts.

The exact experiment used Cell Ontology release 2025-12-17. Its upstream OBO
contains contributor identifiers, so it is not redistributed in this anonymous
snapshot. Download it from:

`http://purl.obolibrary.org/obo/cl/releases/2025-12-17/cl.obo`

Save it as `data/cl.obo` and verify SHA-256:

`2c75f1ce2fef3eae0f5c232647b28726b783fb9970c56657cc5d48769b5966cb`

Compile it with:

```bash
python scolar/compile_ontology.py --input data/cl.obo --output data/cl.pt
```

The generated `.h5ad`, `cl.obo`, and `cl.pt` files are ignored by the package
manifest and should remain local.
