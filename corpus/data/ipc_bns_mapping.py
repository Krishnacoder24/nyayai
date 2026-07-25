"""
Loads the IPC->BNS section concordance table from the versioned JSON
data file and exposes it as the simple {ipc_number: bns_number(s)} dict
ipc.py's existing code already expects (metadata["replaced_by"] =
IPC_TO_BNS[entry["number"]]) - no changes needed in ipc.py itself.

Per this project's rule against fabricated legal mappings, the full
provenance (source, date fetched, caveats) lives in the JSON file
itself under "_provenance" - see that file before trusting this data
for anything production-facing. This table is sourced from a
practitioner-compiled comparative table, not the raw MHA/Gazette
document directly; it's been spot-checked against known reference
mappings but not independently verified section-by-section against
the primary source.

Sections with NO BNS successor (repealed with nothing replacing them,
e.g. IPC 161-165A, folded into the Prevention of Corruption Act
decades before BNS existed) are simply absent from this dict, same as
before - `if entry["number"] in IPC_TO_BNS` correctly stays False for
them rather than pointing at a fabricated placeholder.

Some IPC sections map to MULTIPLE BNS provisions (e.g. section 10
"Man, Woman" splits into BNS 2(19) and 2(35)) - IPC_TO_BNS stores a
list for those instead of a bare string. Callers that assume a single
string should check `isinstance(value, list)` if they need to handle
this - ipc.py's current usage just stores whatever value it gets
straight into metadata, which is agnostic to either shape already.
"""

import json
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "ipc_to_bns_mapping.json"


def _load() -> dict[str, str | list[str]]:
    with open(_DATA_PATH) as f:
        data = json.load(f)

    result: dict[str, str | list[str]] = {}
    for entry in data["mappings"]:
        targets = entry["bns"]
        if not targets:
            continue
        if len(targets) == 1:
            result[entry["ipc_section"]] = targets[0]["section"]
        else:
            result[entry["ipc_section"]] = [t["section"] for t in targets]
    return result


IPC_TO_BNS = _load()
