from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "mindforge_phantom_unicorn.py"


def test_phantom_has_local_rehearsal_control_without_changing_lsl_contract():
    src = SCRIPT.read_text(encoding="utf-8")
    assert '--control-host", default="127.0.0.1"' in src
    assert '--control-port", type=int, default=19744' in src
    assert "udp_command_reader" in src
    assert 'command == "1"' in src and 'set_attention(10.0' in src
    assert 'command == "2"' in src and 'set_attention(12.0' in src
    assert 'command == "0"' in src and "set_attention(None)" in src
    assert 'command.startswith("silence:")' in src
    assert 'command.startswith("gain:")' in src
    assert 'StreamInfo(args.name, "EEG"' in src
    assert 'append_child_value("unit", "microvolts")' in src
