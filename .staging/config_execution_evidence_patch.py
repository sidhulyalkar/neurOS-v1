from pathlib import Path

path = Path("packages/neuros-core/src/neuros/runtime/executor.py")
text = path.read_text(encoding="utf-8")

import_anchor = "from neuros.runtime._validation import positive_finite_real\n"
import_line = (
    "from neuros.runtime._execution_evidence import (\n"
    "    capture_execution_authority,\n"
    "    execution_authority_snapshot,\n"
    ")\n"
)
if import_line not in text:
    if text.count(import_anchor) != 1:
        raise SystemExit("executor import anchor missing or ambiguous")
    text = text.replace(import_anchor, import_line + import_anchor, 1)

capture_anchor = "        graph.validate()\n        self.graph = graph\n"
capture_replacement = (
    "        graph.validate()\n"
    "        self._execution_authority = capture_execution_authority(graph.nodes)\n"
    "        self.graph = graph\n"
)
if capture_replacement not in text:
    if text.count(capture_anchor) != 1:
        raise SystemExit("executor authority-capture anchor missing or ambiguous")
    text = text.replace(capture_anchor, capture_replacement, 1)

snapshot_anchor = '            "edges": edge_metrics,\n            "process_execution": {\n'
snapshot_replacement = (
    '            "edges": edge_metrics,\n'
    '            "execution": execution_authority_snapshot(self._execution_authority),\n'
    '            "process_execution": {\n'
)
if snapshot_replacement not in text:
    if text.count(snapshot_anchor) != 1:
        raise SystemExit("executor snapshot anchor missing or ambiguous")
    text = text.replace(snapshot_anchor, snapshot_replacement, 1)

path.write_text(text, encoding="utf-8")
