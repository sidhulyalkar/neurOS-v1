from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = "packages/neuros-mechint/src/neuros_mechint/representations/cases.py"


def replace_once(old: str, new: str) -> None:
    target = ROOT / PATH
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{PATH}: expected exactly one replacement, found {count}")
    target.write_text(text.replace(old, new, 1))


replace_once(
    '''from .contracts import (
    FitRegime,
''',
    '''from .contracts import (
    EvaluationScope,
    FitRegime,
''',
)
replace_once(
    '''    _freeze_metadata,
    _validated_array,
)
''',
    '''    _freeze_metadata,
    _strict_metric_value,
    _validated_array,
)
''',
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
    '''        regime = FitRegime(self.fit_regime)
        status = CaseStatus(self.status)
''',
    '''        regime = FitRegime(self.fit_regime)
        evaluation_scope = EvaluationScope(self.evaluation_scope)
        status = CaseStatus(self.status)
''',
)
replace_once(
    '''                else:
                    numeric = float(value)
                    if not np.isfinite(numeric):
                        raise ValueError("metric values must be finite or None")
                    metric_values[key] = numeric

        object.__setattr__(self, "fit_regime", regime)
        object.__setattr__(self, "status", status)
''',
    '''                else:
                    metric_values[key] = _strict_metric_value(
                        value,
                        name=f"metric {key!r}",
                    )

        object.__setattr__(self, "fit_regime", regime)
        object.__setattr__(self, "evaluation_scope", evaluation_scope)
        object.__setattr__(self, "status", status)
''',
)
replace_once(
    '''    fit_regime: FitRegime
    total_cases: int
''',
    '''    fit_regime: FitRegime
    evaluation_scope: EvaluationScope
    total_cases: int
''',
)
replace_once(
    '''    metrics: Mapping[str, float | None]
    metadata: Mapping[str, Any] | None = None
''',
    '''    metrics: Mapping[str, float | None]
    metric_n: Mapping[str, int]
    metadata: Mapping[str, Any] | None = None
''',
)
replace_once(
    '''            else:
                numeric = float(value)
                if not np.isfinite(numeric):
                    raise ValueError("summary metric values must be finite or None")
                metric_values[key] = numeric

        object.__setattr__(self, "fit_regime", FitRegime(self.fit_regime))
''',
    '''            else:
                metric_values[key] = _strict_metric_value(
                    value,
                    name=f"summary metric {key!r}",
                )

        metric_n: dict[str, int] = {}
        for key, value in dict(self.metric_n).items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("metric_n IDs must be nonblank strings")
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError("metric_n values must be integers")
            numeric = int(value)
            if numeric < 0 or numeric > counts[1]:
                raise ValueError("metric_n values must be between zero and ok_cases")
            metric_n[key] = numeric
        if set(metric_n) != set(metric_values):
            raise ValueError("metric_n keys must exactly match summary metric keys")

        object.__setattr__(self, "fit_regime", FitRegime(self.fit_regime))
        object.__setattr__(
            self,
            "evaluation_scope",
            EvaluationScope(self.evaluation_scope),
        )
''',
)
replace_once(
    '''        object.__setattr__(self, "metrics", MappingProxyType(metric_values))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def failure_rate(self) -> float:
        return float((self.total_cases - self.ok_cases) / self.total_cases)
''',
    '''        object.__setattr__(self, "metrics", MappingProxyType(metric_values))
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
    '''        if not evaluation_ids:
            raise ValueError("evaluation_sequence_ids cannot be empty")
        if not method_ids:
            raise ValueError("method_ids cannot be empty")
''',
    '''        if not train_ids:
            raise ValueError("train_sequence_ids cannot be empty")
        if not evaluation_ids:
            raise ValueError("evaluation_sequence_ids cannot be empty")
        if not method_ids:
            raise ValueError("method_ids cannot be empty")
        if len(set(train_ids)) != len(train_ids):
            raise ValueError("train sequence IDs must be unique")
''',
)
replace_once(
    '''        if any(not isinstance(value, str) or not value.strip() for value in evaluation_ids):
            raise ValueError("evaluation sequence IDs must be nonblank strings")
''',
    '''        if any(not isinstance(value, str) or not value.strip() for value in train_ids):
            raise ValueError("train sequence IDs must be nonblank strings")
        if any(not isinstance(value, str) or not value.strip() for value in evaluation_ids):
            raise ValueError("evaluation sequence IDs must be nonblank strings")
''',
)
replace_once(
    '''        regimes: dict[str, FitRegime] = {}
        for case in cases:
''',
    '''        regimes: dict[str, FitRegime] = {}
        scopes: dict[str, EvaluationScope] = {}
        for case in cases:
''',
)
replace_once(
    '''            existing = regimes.setdefault(case.method_id, case.fit_regime)
            if existing is not case.fit_regime:
                raise ValueError("all cases for a method must share one fit regime")
''',
    '''            existing = regimes.setdefault(case.method_id, case.fit_regime)
            if existing is not case.fit_regime:
                raise ValueError("all cases for a method must share one fit regime")
            existing_scope = scopes.setdefault(case.method_id, case.evaluation_scope)
            if existing_scope is not case.evaluation_scope:
                raise ValueError("all cases for a method must share one evaluation scope")
''',
)
replace_once(
    '''        regime = cases[0].fit_regime
        counts = {status: 0 for status in CaseStatus}
        metric_values: dict[str, list[float]] = {}
        metric_ids: set[str] = set()
        for case in cases:
            counts[case.status] += 1
            if case.status is not CaseStatus.OK:
                continue
            metric_ids.update(case.metrics)
            for key, value in case.metrics.items():
                if value is not None:
                    metric_values.setdefault(key, []).append(float(value))
        aggregated = {
            key: float(np.mean(metric_values[key])) if metric_values.get(key) else None
            for key in sorted(metric_ids)
        }
        total = len(cases)
        ok = counts[CaseStatus.OK]
        return MethodCaseSummary(
            method_id=method_id,
            fit_regime=regime,
            total_cases=total,
            ok_cases=ok,
            failed_cases=counts[CaseStatus.FAILED],
            unavailable_cases=counts[CaseStatus.UNAVAILABLE],
            nonconverged_cases=counts[CaseStatus.NONCONVERGED],
            metrics=aggregated,
            metadata={
                "aggregation_basis": "successful_cases_with_explicit_denominator",
                "successful_metric_cases": ok,
                "declared_total_cases": total,
            },
        )
''',
    '''        regime = cases[0].fit_regime
        evaluation_scope = cases[0].evaluation_scope
        counts = {status: 0 for status in CaseStatus}
        metric_values: dict[str, list[float]] = {}
        metric_schema: tuple[str, ...] | None = None
        for case in cases:
            counts[case.status] += 1
            if case.status is not CaseStatus.OK:
                continue
            case_schema = tuple(sorted(case.metrics))
            if metric_schema is None:
                metric_schema = case_schema
            elif case_schema != metric_schema:
                raise ValueError(
                    "successful cases for one method must expose an identical metric schema"
                )
            for key, value in case.metrics.items():
                if value is not None:
                    metric_values.setdefault(key, []).append(value)
        metric_schema = metric_schema or ()
        aggregated = {
            key: float(np.mean(metric_values[key])) if metric_values.get(key) else None
            for key in metric_schema
        }
        metric_n = {key: len(metric_values.get(key, ())) for key in metric_schema}
        total = len(cases)
        ok = counts[CaseStatus.OK]
        return MethodCaseSummary(
            method_id=method_id,
            fit_regime=regime,
            evaluation_scope=evaluation_scope,
            total_cases=total,
            ok_cases=ok,
            failed_cases=counts[CaseStatus.FAILED],
            unavailable_cases=counts[CaseStatus.UNAVAILABLE],
            nonconverged_cases=counts[CaseStatus.NONCONVERGED],
            metrics=aggregated,
            metric_n=metric_n,
            metadata={
                "aggregation_basis": "successful_cases_with_per_metric_denominator",
                "successful_metric_cases": ok,
                "declared_total_cases": total,
            },
        )
''',
)
replace_once(
    '''        if len(set(ids)) != len(ids):
            raise ValueError("representation method IDs must be unique")
        self.methods = methods
''',
    '''        if len(set(ids)) != len(ids):
            raise ValueError("representation method IDs must be unique")
        for method in methods:
            try:
                EvaluationScope(method.evaluation_scope)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"representation method {method.method_id!r} must declare a valid "
                    "evaluation_scope"
                ) from exc
        self.methods = methods
''',
)
replace_once(
    '''            fit_regime=method.fit_regime,
            status=status,
''',
    '''            fit_regime=method.fit_regime,
            evaluation_scope=method.evaluation_scope,
            status=status,
''',
)

old_batch = '''    def _batch_inductive_cases(
        self,
        method: RepresentationMethod,
        train: SequenceBatch,
        evaluation: SequenceBatch,
        reference: SequenceBatch | None,
    ) -> list[RepresentationCaseOutcome]:
        try:
            embedding = method.embed(train, evaluation)
            if embedding.sequence_ids != evaluation.sequence_ids:
                raise ValueError(
                    "representation output sequence identity does not match evaluation batch"
                )
            if len(embedding.sequences) != len(evaluation.sequences):
                raise ValueError("representation output changed evaluation sequence count")
        except RepresentationUnavailableError as exc:
            return [
                self._failure_case(method, sequence_id, CaseStatus.UNAVAILABLE, exc)
                for sequence_id in evaluation.sequence_ids
            ]
        except RepresentationNonconvergenceError as exc:
            return [
                self._failure_case(method, sequence_id, CaseStatus.NONCONVERGED, exc)
                for sequence_id in evaluation.sequence_ids
            ]
        except Exception as exc:
            return [
                self._failure_case(method, sequence_id, CaseStatus.FAILED, exc)
                for sequence_id in evaluation.sequence_ids
            ]

        cases: list[RepresentationCaseOutcome] = []
        for index, (sequence_id, source, latent) in enumerate(
            zip(
                evaluation.sequence_ids,
                evaluation.sequences,
                embedding.sequences,
                strict=True,
            )
        ):
            try:
                if source.shape[0] != latent.shape[0]:
                    raise ValueError(
                        "representation output changed the evaluation timepoint count"
                    )
                reference_sequence = (
                    None if reference is None else reference.sequences[index]
                )
                cases.append(
                    RepresentationCaseOutcome(
                        method_id=method.method_id,
                        sequence_id=sequence_id,
                        fit_regime=method.fit_regime,
                        status=CaseStatus.OK,
                        embedding=latent,
                        metrics=self._metrics(source, latent, reference_sequence),
                        metadata={
                            "metric_scope": (
                                "trajectory_local_rigid_transform_invariant"
                            ),
                            "embedding_metadata": dict(embedding.metadata),
                            "execution_scope": (
                                "single_train_fit_full_evaluation_transform"
                            ),
                        },
                    )
                )
            except Exception as exc:
                cases.append(
                    self._failure_case(method, sequence_id, CaseStatus.FAILED, exc)
                )
        return cases
'''
new_batch = '''    def _batch_inductive_cases(
        self,
        method: RepresentationMethod,
        train: SequenceBatch,
        evaluation: SequenceBatch,
        reference: SequenceBatch | None,
    ) -> list[RepresentationCaseOutcome]:
        try:
            embedding = method.embed(train, evaluation)
            if embedding.sequence_ids != evaluation.sequence_ids:
                raise ValueError(
                    "representation output sequence identity does not match evaluation batch"
                )
            if len(embedding.sequences) != len(evaluation.sequences):
                raise ValueError("representation output changed evaluation sequence count")
            for source, latent in zip(evaluation.sequences, embedding.sequences, strict=True):
                if source.shape[0] != latent.shape[0]:
                    raise ValueError(
                        "representation output changed the evaluation timepoint count"
                    )
        except RepresentationUnavailableError as exc:
            return [self._failure_case(method, sequence_id, CaseStatus.UNAVAILABLE, exc) for sequence_id in evaluation.sequence_ids]
        except RepresentationNonconvergenceError as exc:
            return [self._failure_case(method, sequence_id, CaseStatus.NONCONVERGED, exc) for sequence_id in evaluation.sequence_ids]
        except Exception as exc:
            return [self._failure_case(method, sequence_id, CaseStatus.FAILED, exc) for sequence_id in evaluation.sequence_ids]

        cases: list[RepresentationCaseOutcome] = []
        for index, (sequence_id, source, latent) in enumerate(
            zip(evaluation.sequence_ids, evaluation.sequences, embedding.sequences, strict=True)
        ):
            reference_sequence = None if reference is None else reference.sequences[index]
            metrics = self._metrics(source, latent, reference_sequence)
            cases.append(
                RepresentationCaseOutcome(
                    method_id=method.method_id,
                    sequence_id=sequence_id,
                    fit_regime=method.fit_regime,
                    evaluation_scope=method.evaluation_scope,
                    status=CaseStatus.OK,
                    embedding=latent,
                    metrics=metrics,
                    metadata={
                        "metric_scope": "trajectory_local_rigid_transform_invariant",
                        "embedding_metadata": dict(embedding.metadata),
                        "evaluation_scope": EvaluationScope(method.evaluation_scope).value,
                    },
                )
            )
        return cases
'''
replace_once(old_batch, new_batch)

old_local = '''    def _sequence_local_cases(
        self,
        method: RepresentationMethod,
        train: SequenceBatch,
        evaluation: SequenceBatch,
        reference: SequenceBatch | None,
    ) -> list[RepresentationCaseOutcome]:
        cases: list[RepresentationCaseOutcome] = []
        for index, sequence_id in enumerate(evaluation.sequence_ids):
            evaluation_case = self._single_batch(evaluation, index)
            try:
                embedding = method.embed(train, evaluation_case)
                if (
                    embedding.sequence_ids != (sequence_id,)
                    or len(embedding.sequences) != 1
                ):
                    raise ValueError(
                        "sequence-local representation output identity does not match case"
                    )
                latent = embedding.sequences[0]
                source = evaluation.sequences[index]
                if source.shape[0] != latent.shape[0]:
                    raise ValueError(
                        "representation output changed the evaluation timepoint count"
                    )
                reference_sequence = (
                    None if reference is None else reference.sequences[index]
                )
                cases.append(
                    RepresentationCaseOutcome(
                        method_id=method.method_id,
                        sequence_id=sequence_id,
                        fit_regime=method.fit_regime,
                        status=CaseStatus.OK,
                        embedding=latent,
                        metrics=self._metrics(source, latent, reference_sequence),
                        metadata={
                            "metric_scope": (
                                "trajectory_local_rigid_transform_invariant"
                            ),
                            "embedding_metadata": dict(embedding.metadata),
                            "execution_scope": (
                                "preserved_sequence_local_fit_or_lookup"
                            ),
                        },
                    )
                )
            except RepresentationUnavailableError as exc:
                cases.append(
                    self._failure_case(
                        method, sequence_id, CaseStatus.UNAVAILABLE, exc
                    )
                )
            except RepresentationNonconvergenceError as exc:
                cases.append(
                    self._failure_case(
                        method, sequence_id, CaseStatus.NONCONVERGED, exc
                    )
                )
            except Exception as exc:
                cases.append(
                    self._failure_case(method, sequence_id, CaseStatus.FAILED, exc)
                )
        return cases
'''
new_local = '''    def _sequence_local_cases(
        self,
        method: RepresentationMethod,
        train: SequenceBatch,
        evaluation: SequenceBatch,
        reference: SequenceBatch | None,
    ) -> list[RepresentationCaseOutcome]:
        cases: list[RepresentationCaseOutcome] = []
        for index, sequence_id in enumerate(evaluation.sequence_ids):
            evaluation_case = self._single_batch(evaluation, index)
            try:
                embedding = method.embed(train, evaluation_case)
                if embedding.sequence_ids != (sequence_id,) or len(embedding.sequences) != 1:
                    raise ValueError(
                        "sequence-local representation output identity does not match case"
                    )
                latent = embedding.sequences[0]
                source = evaluation.sequences[index]
                if source.shape[0] != latent.shape[0]:
                    raise ValueError(
                        "representation output changed the evaluation timepoint count"
                    )
            except RepresentationUnavailableError as exc:
                cases.append(self._failure_case(method, sequence_id, CaseStatus.UNAVAILABLE, exc))
                continue
            except RepresentationNonconvergenceError as exc:
                cases.append(self._failure_case(method, sequence_id, CaseStatus.NONCONVERGED, exc))
                continue
            except Exception as exc:
                cases.append(self._failure_case(method, sequence_id, CaseStatus.FAILED, exc))
                continue

            reference_sequence = None if reference is None else reference.sequences[index]
            metrics = self._metrics(source, latent, reference_sequence)
            cases.append(
                RepresentationCaseOutcome(
                    method_id=method.method_id,
                    sequence_id=sequence_id,
                    fit_regime=method.fit_regime,
                    evaluation_scope=method.evaluation_scope,
                    status=CaseStatus.OK,
                    embedding=latent,
                    metrics=metrics,
                    metadata={
                        "metric_scope": "trajectory_local_rigid_transform_invariant",
                        "embedding_metadata": dict(embedding.metadata),
                        "evaluation_scope": EvaluationScope(method.evaluation_scope).value,
                    },
                )
            )
        return cases
'''
replace_once(old_local, new_local)

replace_once(
    '''        for method in self.methods:
            regime = FitRegime(method.fit_regime)
            if regime is FitRegime.TRAIN_ONLY_INDUCTIVE:
                cases.extend(
                    self._batch_inductive_cases(
                        method, train, evaluation, reference
                    )
                )
            else:
                cases.extend(
                    self._sequence_local_cases(
                        method, train, evaluation, reference
                    )
                )
''',
    '''        for method in self.methods:
            scope = EvaluationScope(method.evaluation_scope)
            if scope is EvaluationScope.BATCH_TRANSFORM:
                cases.extend(self._batch_inductive_cases(method, train, evaluation, reference))
            elif scope is EvaluationScope.SEQUENCE_LOCAL:
                cases.extend(self._sequence_local_cases(method, train, evaluation, reference))
            else:  # pragma: no cover
                raise ValueError(f"unsupported evaluation scope {scope!r}")
''',
)
replace_once(
    '''                "case_authority": (
                    "complete_method_x_sequence_cartesian_product"
                ),
''',
    '''                "case_authority": (
                    "complete_method_x_sequence_cartesian_product"
                ),
                "evaluation_scope_authority": (
                    "explicit_method_declared_batch_or_sequence_local"
                ),
''',
)

print("case semantics patch applied")
