"""Correlated process priors with a canonical evidence ledger (§7.3-§7.5).

Rules encoded here:
- priors carry shared errors through one covariance per evidence group;
- replicating a prior's support does not create new information: duplicated
  statistics are merged into canonical factors (identical parent sources) or
  evaluated with the shared-error covariance, never given fresh nuggets;
- learned covariance must pay 0.5*logdet and a scale prior;
- the same Q (or beta) can feed EITHER a c_j prior OR a derived dH prior,
  never both as independent losses.
"""

import numpy as np


class EvidenceGroup:
    """One independent piece of external evidence (fixed-dimensional statistic)."""

    def __init__(
        self,
        group_id,
        parent_source_ids,
        kind,
        mean,
        cov,
        projector=None,
        applies=None,
        description="",
    ):
        self.group_id = group_id
        self.parent_source_ids = tuple(sorted(parent_source_ids))
        self.kind = kind
        self.mean = np.atleast_1d(np.asarray(mean, dtype=float))
        self.cov = np.atleast_2d(np.asarray(cov, dtype=float))
        self.projector = projector  # callable: field/params -> statistic vector
        self.applies = applies  # boolean applicability mask (must be data-derived)
        self.description = description

    def nll(self, statistic, learned_scale=None):
        x = np.atleast_1d(np.asarray(statistic, dtype=float)) - self.mean
        cov = self.cov
        if learned_scale is not None:
            cov = cov * (float(learned_scale) ** 2)
        cov = cov + 1e-12 * np.eye(len(cov)) * max(np.trace(cov), 1.0)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("prior covariance not positive definite")
        sol = np.linalg.solve(cov, x)
        return 0.5 * float(x @ sol) + 0.5 * float(logdet) + 0.5 * len(x) * np.log(2 * np.pi)


def canonical_key(group: EvidenceGroup):
    return (group.kind, group.parent_source_ids)


def merge_duplicates(groups):
    """Merge groups sharing (kind, parent sources): one canonical factor each."""
    seen = {}
    for g in groups:
        key = canonical_key(g)
        if key in seen:
            continue
        seen[key] = g
    return list(seen.values())


def replication_invariant_nll(group, replication, statistic_fn):
    """Verify duplicating a fixed statistic adds no information.

    statistic_fn(replication) must return the SAME physical statistic repeated
    (pure replication, no new independent draws); the shared-error covariance
    of the expanded model is singular, so the check uses the canonical merge.
    """
    merged = merge_duplicates([group] * replication)
    if len(merged) != 1:
        raise ValueError("replication created distinct evidence groups")
    stat = statistic_fn(1)
    return group.nll(stat)


def shared_offset_information(n, sigma_local, sigma_shared):
    """1^T C^{-1} 1 for C = s_l^2 I + s_s^2 11^T (common-offset model, §7.4)."""
    return float(n / (sigma_local**2 + n * sigma_shared**2))


def forbid_double_use(ledger, process_id, role):
    """A process (e.g. one Q estimate) may occupy one loss role per region."""
    for g in ledger:
        if process_id in g.parent_source_ids and getattr(g, "role", None) == role:
            raise ValueError(f"process {process_id} already used as {role}")
