"""
Loads the CrPC->BNSS section concordance table from the versioned JSON
data file, exposed as the simple {crpc_number: bnss_number} dict shape -
same contract as corpus/data/ipc_bns_mapping.py's IPC_TO_BNS, so a future
crpc.py parser can wire it in the same way ipc.py already does
(metadata["replaced_by"] = CRPC_TO_BNSS[entry["number"]]).

NOTE: unlike IPC_TO_BNS, nothing in this codebase currently imports this
- there's no corpus/parsers/crpc.py yet. Sourced and stored now per issue
#32's task, ready for whenever a CrPC parser exists.

Per this project's rule against fabricated legal mappings, full
provenance (source, date fetched, caveats) lives in the JSON file's
"_provenance" field - see that before trusting this data for anything
production-facing. Same practitioner-compiled-table caveat as
ipc_bns_mapping.py applies here.

Unlike the IPC table, every mapped CrPC section here has exactly ONE
BNSS target (no multi-target list values) - the source table didn't
have any one-to-many splits the way a few IPC sections did.
"""

import json
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "crpc_to_bnss_mapping.json"


def _load() -> dict[str, str]:
    with open(_DATA_PATH) as f:
        data = json.load(f)

    return {
        entry["crpc"]: entry["bnss"]
        for entry in data["mappings"]
        if entry["bnss"] is not None
    }


CRPC_TO_BNSS = _load()
