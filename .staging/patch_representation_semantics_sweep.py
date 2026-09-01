from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = "packages/neuros-mechint/src/neuros_mechint/representations/sweep.py"


def replace_once(old: str, new: str) -> None:
    target = ROOT / PATH
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{PATH}: expected exactly one replacement, found {count}")
    target.write_text(text.replace(old, new, 1))


replace_once(
    "from .contracts import FitRegime, RepresentationMethod, _freeze_metadata\n",
    "from .contracts import (\n    EvaluationScope,\n    FitRegime,\n    RepresentationMethod,\n    _freeze_metadata,\n    _strict_metric_value,\n)\n",
)
replace_once(
    '''    fit_regime: FitRegime
    status: CaseStatus
''',
    '''    fit_regime: FitRegime
    evaluation_scope: EvaluationScope
    status: CaseStatus
''',
)
replace_once(
    '''        status = CaseStatus(self.status)
        regime = FitRegime(self.fit_regime)
''',
    '''        status = CaseStatus(self.status)
        regime = FitRegime(self.fit_regime)
        evaluation_scope = EvaluationScope(self.evaluation_scope)
''',
)
replace_once(
    '''                else:
                    numeric = float(value)
                    if not np.isfinite(numeric):
                        raise ValueError("metric values must be finite or None")
                    metric_values[key] = numeric
''',
    '''                else:
                    metric_values[key] = _strict_metric_value(
                        value,
                        name=f"metric {key!r}",
                    )
''',
)
replace_once(
    '''        object.__setattr__(self, "fit_regime", regime)
        object.__setattr__(self, "status", status)
''',
    '''        object.__setattr__(self, "fit_regime", regime)
        object.__setattr__(self, "evaluation_scope", evaluation_scope)
        object.__setattr__(self, "status", status)
''',
)
replace_once(
    '''        else:
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError("metric summary values must be finite or None")
            output[key] = numeric
''',
    '''        else:
            output[key] = _strict_metric_value(
                value,
                name=f"metric summary {key!r}",
            )
''',
)
replace_once(
    '''    fit_regime: FitRegime
    noise_std: float
''',
    '''    fit_regime: FitRegime
    evaluation_scope: EvaluationScope
    noise_std: float
''',
)
replace_once(
    '''    metric_sem: Mapping[str, float | None]
    metadata: Mapping[str, Any] | None = None
''',
    '''    metric_sem: Mapping[str, float | None]
    metric_n: Mapping[str, int]
    metadata: Mapping[str, Any] | None = None
''',
)
replace_once(
    '''        object.__setattr__(self, "noise_std", noise)
        object.__setattr__(self, "fit_regime", FitRegime(self.fit_regime))
''',
    '''        object.__setattr__(self, "noise_std", noise)
        object.__setattr__(self, "fit_regime", FitRegime(self.fit_regime))
        object.__setattr__(
            self,
            "evaluation_scope",
            EvaluationScope(self.evaluation_scope),
        )
''',
)
replace_once(
    '''        object.__setattr__(
            self, "metric_sem", _finite_metric_mapping(self.metric_sem)
        )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def failure_rate(self) -> float:
        return float((self.total_cases - self.ok_cases) / self.total_cases)
''',
    '''        object.__setattr__(
            self, "metric_sem", _finite_metric_mapping(self.metric_sem)
        )
        metric_n: dict[str, int] = {}
        for key, value in dict(self.metric_n).items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("metric_n IDs must be nonblank strings")
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError("metric_n values must be integers")
            numeric = int(value)
            if numeric < 0 or numeric > self.ok_cases:
                raise ValueError("metric_n values must be between zero and ok_cases")
            metric_n[key] = numeric
        expected_metric_keys = set(self.metric_mean)
        if set(self.metric_std) != expected_metric_keys:
            raise ValueError("metric_std keys must exactly match metric_mean keys")
        if set(self.metric_sem) != expected_metric_keys:
            raise ValueError("metric_sem keys must exactly match metric_mean keys")
        if set(metric_n) != expected_metric_keys:
            raise ValueError("metric_n keys must exactly match metric_mean keys")
        object.__setattr__(self, "metric_n", MappingProxyType(metric_n))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def non_ok_rate(self) -> float:
        return float((self.total_cases - self.ok_cases) / self.total_cases)

    @property
    def failed_rate(self) -> float:
        return float(self.failed_cases / self.total_cases)

    @property
    def unavailable_rate(self) -> float:
        return float(self.unavailable_cases / self.total_cases)

    @property
    def nonconverged_rate(self) -> float:
        return float(self.nonconverged_cases / self.total_cases)

    @property
    def failure_rate(self) -> float:
        """Deprecated compatibility alias for the broader non-ok rate."""
        return self.non_ok_rate
''',
)
replace_once(
    '''        regimes: dict[str, FitRegime] = {}
        for record in records:
''',
    '''        regimes: dict[str, FitRegime] = {}
        scopes: dict[str, EvaluationScope] = {}
        for record in records:
''',
)
replace_once(
    '''            existing = regimes.setdefault(record.method_id, record.fit_regime)
            if existing is not record.fit_regime:
                raise ValueError("fit regime changed across sweep points")
''',
    '''            existing = regimes.setdefault(record.method_id, record.fit_regime)
            if existing is not record.fit_regime:
                raise ValueError("fit regime changed across sweep points")
            existing_scope = scopes.setdefault(record.method_id, record.evaluation_scope)
            if existing_scope is not record.evaluation_scope:
                raise ValueError("evaluation scope changed across sweep points")
''',
)
replace_once(
    '''        counts = {status: 0 for status in CaseStatus}
        values: dict[str, list[float]] = {}
        metric_ids: set[str] = set()
        for record in records:
            counts[record.status] += 1
            if record.status is not CaseStatus.OK:
                continue
            metric_ids.update(record.metrics)
            for key, value in record.metrics.items():
                if value is not None:
                    values.setdefault(key, []).append(float(value))

        means: dict[str, float | None] = {}
        stds: dict[str, float | None] = {}
        sems: dict[str, float | None] = {}
        for key in sorted(metric_ids):
''',
    '''        counts = {status: 0 for status in CaseStatus}
        values: dict[str, list[float]] = {}
        metric_schema: tuple[str, ...] | None = None
        for record in records:
            counts[record.status] += 1
            if record.status is not CaseStatus.OK:
                continue
            record_schema = tuple(sorted(record.metrics))
            if metric_schema is None:
                metric_schema = record_schema
            elif record_schema != metric_schema:
                raise ValueError(
                    "successful sweep records must expose an identical metric schema"
                )
            for key, value in record.metrics.items():
                if value is not None:
                    values.setdefault(key, []).append(value)

        means: dict[str, float | None] = {}
        stds: dict[str, float | None] = {}
        sems: dict[str, float | None] = {}
        metric_n: dict[str, int] = {}
        for key in metric_schema or ():
''',
)
replace_once(
    '''            if samples.size == 0:
                means[key] = stds[key] = sems[key] = None
                continue
            means[key] = float(np.mean(samples))
''',
    '''            metric_n[key] = int(samples.size)
            if samples.size == 0:
                means[key] = stds[key] = sems[key] = None
                continue
            means[key] = float(np.mean(samples))
''',
)
replace_once(
    '''            method_id=method_id,
            fit_regime=records[0].fit_regime,
            noise_std=noise_std,
''',
    '''            method_id=method_id,
            fit_regime=records[0].fit_regime,
            evaluation_scope=records[0].evaluation_scope,
            noise_std=noise_std,
''',
)
replace_once(
    '''            metric_mean=means,
            metric_std=stds,
            metric_sem=sems,
            metadata={
''',
    '''            metric_mean=means,
            metric_std=stds,
            metric_sem=sems,
            metric_n=metric_n,
            metadata={
''',
)
replace_once(
    '''    declared_regimes: dict[str, FitRegime] | None = None
    evaluation_ids: tuple[str, ...] | None = None
''',
    '''    declared_regimes: dict[str, FitRegime] | None = None
    declared_scopes: dict[str, EvaluationScope] | None = None
    evaluation_ids: tuple[str, ...] | None = None
''',
)
replace_once(
    '''            regimes = {
                method.method_id: FitRegime(method.fit_regime) for method in methods
            }
            if declared_method_ids is None:
                declared_method_ids = method_ids
                declared_regimes = regimes
            elif method_ids != declared_method_ids or regimes != declared_regimes:
                raise ValueError(
                    "method_factory must return the same ordered method IDs and fit "
                    "regimes at every sweep point"
                )
''',
    '''            regimes = {
                method.method_id: FitRegime(method.fit_regime) for method in methods
            }
            scopes = {
                method.method_id: EvaluationScope(method.evaluation_scope)
                for method in methods
            }
            if declared_method_ids is None:
                declared_method_ids = method_ids
                declared_regimes = regimes
                declared_scopes = scopes
            elif (
                method_ids != declared_method_ids
                or regimes != declared_regimes
                or scopes != declared_scopes
            ):
                raise ValueError(
                    "method_factory must return the same ordered method IDs, fit "
                    "regimes, and evaluation scopes at every sweep point"
                )
''',
)
replace_once(
    '''                        fit_regime=case.fit_regime,
                        status=case.status,
''',
    '''                        fit_regime=case.fit_regime,
                        evaluation_scope=case.evaluation_scope,
                        status=case.status,
''',
)
replace_once(
    '"schema": "neuros.representation.controlled_noise_sweep.v1",',
    '"schema": "neuros.representation.controlled_noise_sweep.v2",',
)

print("sweep semantics patch applied")
