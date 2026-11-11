# JSON-line dispatch mirroring Domicile's _dispatch_to_core: the phase-1
# cutover swaps the store behind this surface and no caller changes.
import json


def _ops(core):
    return {
        "search": core.search,
        "hint": core.hint,
        "add": core.add,
        "add_verbatim": core.add_verbatim,
        "list": core.list_facts,
        "list_facts": core.list_facts,
        "get": core.get,
        "update": core.update,
        "delete": core.delete,
        "mark_superseded": core.mark_superseded,
    }


def handle(core, payload):
    op = payload.get("op")
    fn = _ops(core).get(op)
    if fn is None:
        return {"error": f"unknown op: {op}"}
    args = {k: v for k, v in payload.items() if k != "op"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"bad args for {op}: {e}"}


def handle_line(core, line):
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"bad json: {e}"})
    return json.dumps(handle(core, payload))
