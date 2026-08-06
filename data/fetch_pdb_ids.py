"""
Query RCSB's search API for single-chain protein structures with
50-300 residues, and save the resulting PDB IDs to a text file.

Output of this script is the --id-list input for download_pdbs.py.
"""

import argparse
from pathlib import Path

import requests

# RCSB's structured search API (separate from the file-download endpoint
# used in download_pdbs.py). Takes a JSON query, returns matching entry IDs.
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def build_query(min_length: int, max_length: int, max_results: int) -> dict:
    """Build the RCSB search request body for our filter criteria."""
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    # Only entries that are protein-only (no nucleic acids, no ligand-only entries)
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                        "operator": "exact_match",
                        "value": "Protein (only)",
                    },
                },
                {
                    # Exactly one distinct protein entity -> single chain, not a complex
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "equals",
                        "value": 1,
                    },
                },
                {
                    # Residue count lower bound. Note: polymer_monomer_count (no
                    # min/max suffix) exists in RCSB's schema but isn't search-enabled;
                    # the queryable per-entry fields are the _minimum/_maximum variants.
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_monomer_count_minimum",
                        "operator": "greater_or_equal",
                        "value": min_length,
                    },
                },
                {
                    # Residue count upper bound
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_monomer_count_maximum",
                        "operator": "less_or_equal",
                        "value": max_length,
                    },
                },
            ],
        },
        "return_type": "entry",  # we want PDB entry IDs, not entity/assembly IDs
        "request_options": {
            "paginate": {"start": 0, "rows": max_results},
            "results_content_type": ["experimental"],  # skip computed/predicted models
        },
    }


def fetch_pdb_ids(min_length: int, max_length: int, max_results: int) -> list[str]:
    """Run the search query and return a list of matching PDB IDs."""
    query = build_query(min_length, max_length, max_results)
    response = requests.post(SEARCH_URL, json=query, timeout=30)
    response.raise_for_status()

    data = response.json()
    return [hit["identifier"] for hit in data.get("result_set", [])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-length", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--max-results", type=int, default=1000)
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("data/pdb_ids.txt"),
        help="Where to save the resulting list of PDB IDs.",
    )
    args = parser.parse_args()

    print(f"Querying RCSB for single-chain proteins, {args.min_length}-{args.max_length} residues...")
    pdb_ids = fetch_pdb_ids(args.min_length, args.max_length, args.max_results)
    print(f"Found {len(pdb_ids)} matching entries.")

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text("\n".join(pdb_ids))
    print(f"Saved to {args.out_file}")


if __name__ == "__main__":
    main()
