"""
Download raw .pdb files from RCSB given a list of PDB IDs.
This is step 1 of the structure-tokenizer pipeline.
"""  # module docstring: becomes __doc__, shown by --help later

import argparse          # stdlib: parses command-line flags like --id-list
import time               # stdlib: gives us time.sleep() to throttle requests
from pathlib import Path  # stdlib: object-oriented filesystem paths (vs raw strings)

import requests  # 3rd-party (pip install requests): makes HTTP GET calls

RCSB_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.pdb"  # URL pattern; {pdb_id} filled in later
REQUEST_DELAY_SECONDS = 0.2  # seconds to sleep between downloads, avoids hammering RCSB


def read_pdb_id_list(list_path: Path) -> list[str]:      # takes a Path, returns a list of strings
    """Load PDB IDs from a plain text file, one ID per line."""  # docstring for this function
    with open(list_path) as f:                            # open file; auto-closes when block ends
        ids = [                                            # start building a list via comprehension
            line.strip().upper()                           # per kept line: strip whitespace/newline, uppercase it
            for line in f                                  # iterate the file object one line at a time
            if line.strip()                                # skip blank lines (empty string is falsy)
        ]
    return ids                                              # hand back the cleaned list of IDs


def download_one(pdb_id: str, out_dir: Path, session: requests.Session) -> bool:  # one ID -> True/False
    """Download a single PDB file. Returns True on success, False on failure."""  # docstring
    out_path = out_dir / f"{pdb_id}.pdb"    # Path "/" joins segments -> e.g. data/raw_pdb/1ABC.pdb

    if out_path.exists():        # already downloaded in a previous run?
        return True               # treat as success, skip re-downloading (makes script resumable)

    url = RCSB_URL_TEMPLATE.format(pdb_id=pdb_id)  # substitute {pdb_id} into the URL template
    try:                                             # begin error-handling block
        response = session.get(url, timeout=15)      # HTTP GET; give up after 15s with no response
        response.raise_for_status()                  # raise an exception if status code is 4xx/5xx
    except requests.RequestException as e:            # catch timeout / connection error / bad status
        print(f"  FAILED {pdb_id}: {e}")               # log which ID failed and why
        return False                                    # signal failure to caller

    out_path.write_bytes(response.content)  # write raw response bytes to disk (bytes, not text — avoids encoding issues)
    return True                              # signal success to caller


def main():                                     # entry point, called only when script is run directly
    parser = argparse.ArgumentParser(description=__doc__)  # create CLI parser; reuse module docstring as help text

    parser.add_argument(              # register the --id-list flag
        "--id-list",                   # flag name as typed on the command line
        type=Path,                     # argparse wraps the given string in a Path automatically
        required=True,                 # script errors out if this flag is missing
        help="Text file with one PDB ID per line (e.g. 1ABC).",  # shown in --help
    )
    parser.add_argument(              # register the --out-dir flag
        "--out-dir",                   # flag name
        type=Path,                     # again auto-wrapped as a Path
        default=Path("data/raw_pdb"),  # used if the flag is omitted
        help="Directory to save downloaded .pdb files into.",  # shown in --help
    )
    args = parser.parse_args()  # actually read sys.argv and populate args.id_list / args.out_dir

    args.out_dir.mkdir(parents=True, exist_ok=True)  # create output dir (and any missing parents); no error if it exists
    pdb_ids = read_pdb_id_list(args.id_list)          # load the list of IDs from the given file
    print(f"Downloading {len(pdb_ids)} PDB files to {args.out_dir}/")  # status message to the user

    session = requests.Session()  # reuse one TCP connection across all ~1000 requests instead of opening a new one each time

    failed = []                                    # collect IDs that failed, to report/save at the end
    for i, pdb_id in enumerate(pdb_ids, start=1):   # loop with a 1-based counter i alongside each pdb_id
        ok = download_one(pdb_id, args.out_dir, session)  # attempt the download, get True/False back
        if not ok:                                   # if it failed...
            failed.append(pdb_id)                     # ...record it
        if i % 50 == 0:                              # every 50th item...
            print(f"  {i}/{len(pdb_ids)} processed...")  # ...print a progress update
        time.sleep(REQUEST_DELAY_SECONDS)             # pause briefly before the next request

    print(f"Done. {len(pdb_ids) - len(failed)} succeeded, {len(failed)} failed.")  # final summary
    if failed:                                          # if anything failed...
        failed_path = args.out_dir / "_failed_ids.txt"    # ...build a path for a failure-log file
        failed_path.write_text("\n".join(failed))          # ...write one failed ID per line
        print(f"Failed IDs written to {failed_path}")       # ...tell the user where to find it


if __name__ == "__main__":  # True only when this file is run directly (not imported as a module)
    main()                    # kick off the whole script
