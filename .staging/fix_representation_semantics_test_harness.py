from pathlib import Path

path = Path(__file__).with_name("patch_representation_semantics_tests.py")
text = path.read_text()
old = '''replace_once(
    case_tests,
    '    assert summary.failure_rate == pytest.approx(1 / 3)\\n',
    '    assert summary.non_ok_rate == pytest.approx(1 / 3)\\n'
    '    assert summary.failed_rate == pytest.approx(1 / 3)\\n'
    '    assert summary.unavailable_rate == 0.0\\n',
)
replace_once(
    case_tests,
    '    assert summary.failure_rate == pytest.approx(1 / 3)\\n',
    '    assert summary.non_ok_rate == pytest.approx(1 / 3)\\n'
    '    assert summary.failed_rate == 0.0\\n'
    '    assert summary.unavailable_rate == pytest.approx(1 / 3)\\n',
)
'''
new = '''replace_once(
    case_tests,
    '    assert summary.failed_cases == 1\\n'
    '    assert summary.failure_rate == pytest.approx(1 / 3)\\n'
    '    assert summary.metadata["successful_metric_cases"] == 2\\n',
    '    assert summary.failed_cases == 1\\n'
    '    assert summary.non_ok_rate == pytest.approx(1 / 3)\\n'
    '    assert summary.failed_rate == pytest.approx(1 / 3)\\n'
    '    assert summary.unavailable_rate == 0.0\\n'
    '    assert summary.metadata["successful_metric_cases"] == 2\\n',
)
replace_once(
    case_tests,
    '    assert summary.unavailable_cases == 1\\n'
    '    assert summary.failure_rate == pytest.approx(1 / 3)\\n',
    '    assert summary.unavailable_cases == 1\\n'
    '    assert summary.non_ok_rate == pytest.approx(1 / 3)\\n'
    '    assert summary.failed_rate == 0.0\\n'
    '    assert summary.unavailable_rate == pytest.approx(1 / 3)\\n',
)
'''
count = text.count(old)
if count != 1:
    raise RuntimeError(f"expected one ambiguous harness block, found {count}")
path.write_text(text.replace(old, new, 1))
print("test patch harness markers made context-specific")
