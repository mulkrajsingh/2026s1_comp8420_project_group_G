"""Build open academic + ResearchQA hybrid LoRA JSONL."""

from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timezone

from datasets import concatenate_datasets

from lora_dataset.converters import (
    convert_peerread_rows,
    convert_qasper_rows,
    convert_researchqa_rows,
    convert_scicite_rows,
    convert_scitldr_rows,
    shuffle_records,
)
from lora_dataset.io import validate_messages_record, write_jsonl
from lora_dataset.loaders import (
    load_peerread,
    load_qasper,
    load_researchqa,
    load_scicite,
    load_scitldr,
)
from lora_dataset.paths import (
    HYBRID_JSONL,
    HYBRID_MANIFEST,
    OPEN_ACADEMIC_PER_SOURCE,
    RESEARCHQA_LIMIT,
    SEED,
)


def build_open_hybrid() -> dict[str, int]:
    """Download HF datasets, convert, shuffle, and write hybrid JSONL + manifest."""
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")
    HYBRID_JSONL.parent.mkdir(parents=True, exist_ok=True)

    qasper_all = concatenate_datasets([load_qasper("train"), load_qasper("validation")])
    qasper_records = convert_qasper_rows(qasper_all, limit=OPEN_ACADEMIC_PER_SOURCE, seed=SEED)
    print(f"QASPER rows {len(qasper_records)}")

    scitldr_records = convert_scitldr_rows(
        load_scitldr("train", config="Abstract"), limit=OPEN_ACADEMIC_PER_SOURCE, seed=SEED
    )
    print(f"SciTLDR rows {len(scitldr_records)}")

    scicite_records = convert_scicite_rows(
        load_scicite("train", config="extended"), limit=OPEN_ACADEMIC_PER_SOURCE, seed=SEED
    )
    print(f"SciCite rows {len(scicite_records)}")

    peerread_records = convert_peerread_rows(
        load_peerread("train", config="reviews"), limit=OPEN_ACADEMIC_PER_SOURCE, seed=SEED
    )
    print(f"PeerRead rows {len(peerread_records)}")

    open_academic = qasper_records + scitldr_records + scicite_records + peerread_records
    print(f"Open academic rows {len(open_academic)}", Counter(r["source"] for r in open_academic))

    researchqa_records = convert_researchqa_rows(
        load_researchqa("test"), limit=RESEARCHQA_LIMIT, seed=SEED
    )
    print(f"ResearchQA rows {len(researchqa_records)}")

    hybrid_records = shuffle_records(open_academic + researchqa_records, SEED)
    counts = Counter(r["source"] for r in hybrid_records)
    write_jsonl(HYBRID_JSONL, hybrid_records)

    HYBRID_MANIFEST.write_text(
        "\n".join(
            [
                "# Open Academic + ResearchQA Hybrid LoRA Dataset Manifest",
                "",
                f"Generated: {datetime.now(timezone.utc).isoformat()}",
                f"Seed: {SEED}",
                f"Open academic per-source target: {OPEN_ACADEMIC_PER_SOURCE}",
                f"ResearchQA target: {RESEARCHQA_LIMIT}",
                f"Total rows: {len(hybrid_records)}",
                "",
                "## Rows by source",
                *[f"- {k}: {v}" for k, v in sorted(counts.items())],
                "",
                "Sources: QASPER, SciTLDR, SciCite, PeerRead, ResearchQA (see lora_dataset/README.md).",
                "",
            ]
        ),
        encoding="utf-8",
    )

    sample = hybrid_records[0]
    validate_messages_record(sample)
    print(f"Wrote {len(hybrid_records)} hybrid rows to {HYBRID_JSONL}")
    return dict(counts)
