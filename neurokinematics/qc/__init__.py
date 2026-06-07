"""Data quality-control checks for neurokinematics sessions, subjects, groups.

The QC layer is deliberately GUI-independent and side-effect free: each check
reads a session's recorded outputs and returns a structured result, so it can be
run from the GUI, a script, a notebook, or CI.

    from neurokinematics.qc import run_qc
    report = run_qc(session)        # or a subject / group
    print(report.status)            # PASS / WARN / FAIL
    report.to_dict()                # JSON-serialisable
"""

from neurokinematics.qc.session_qc import (
    QCStatus,
    QCResult,
    QCReport,
    run_qc,
    run_session_qc,
    run_subject_qc,
    run_group_qc,
)

__all__ = [
    "QCStatus",
    "QCResult",
    "QCReport",
    "run_qc",
    "run_session_qc",
    "run_subject_qc",
    "run_group_qc",
]
