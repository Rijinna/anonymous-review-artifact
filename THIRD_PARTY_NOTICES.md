# Third-party materials

## Cell Ontology

The experiments use a snapshot of the [Cell Ontology](https://obofoundry.org/ontology/cl.html) whose file header reports release date 2025-12-17. The OBO file is not bundled in this anonymous snapshot because its upstream metadata contains contributor identifiers. Exact download and checksum instructions are in `data/README.md`.

Cell Ontology is provided by its contributors under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/). The ontology is third-party material and is not covered by this repository's MIT License.

The code converts the downloaded OBO file into a PyTorch artifact (`data/cl.pt`) at runtime. Any use or redistribution of the ontology remains subject to its own license and attribution requirements.
