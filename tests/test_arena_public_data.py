from __future__ import annotations

from neuros.arena.public_data import iter_moabb_raw_runs


class FakeRaw:
    ch_names = ["Oz"]
    info = {"sfreq": 250.0}

    def get_data(self):
        return None


class FakeDataset:
    code = "FakeSSVEP"

    def get_data(self, subjects=None):
        assert subjects == [1, 2]
        return {
            1: {"0": {"0": FakeRaw(), "1": FakeRaw()}},
            2: {"0": {"0": FakeRaw()}},
        }


def test_moabb_adapter_uses_documented_subject_session_run_structure():
    domains = list(iter_moabb_raw_runs(FakeDataset(), subjects=[1, 2]))
    assert len(domains) == 3
    assert domains[0].dataset == "FakeSSVEP"
    assert domains[0].subject == "1"
    assert domains[0].session == "0"
    assert domains[0].run == "0"
    assert domains[0].domain_id == "FakeSSVEP:sub-1:ses-0:run-0"
    assert domains[-1].domain_id == "FakeSSVEP:sub-2:ses-0:run-0"
