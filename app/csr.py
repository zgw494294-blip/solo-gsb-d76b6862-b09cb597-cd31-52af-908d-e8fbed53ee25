"""PKCS#10 CSR intake for POST /v1/issuance/check-csr.

Parses a PEM-encoded certificate signing request with ``cryptography``,
verifies its self-signature, and extracts the identifiers to authorize
solely from the subjectAltName extension — the subject CN is never
consulted as a fallback. Every SAN entry must be a DNSName; any other
GeneralName type (IP address, email, URI, ...) fails the request.

Each DNS identifier is normalized with the same rules as every other
name in the service (case and a single trailing dot are ignored). A
wildcard identifier is only ``*.`` as the left-most label; for the CAA
lookup the prefix is removed and ``wildcard=True`` is passed, per the
wildcard issuance rules. Any other ``*`` placement, or any name the
base normalizer rejects, is a 422-grade error, as are identifiers that
collide after normalization.

All identifiers are then authorized in SAN order against one shared
prepared snapshot of the caller-supplied record set (a single
:func:`prepare_records` pass per request); alias errors (CNAME loop,
>8 hops) still surface as 422. The CSR as a whole is acceptable only
when every identifier is authorized.
"""

from __future__ import annotations

import hashlib

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import ExtensionOID

from .engine import (
    CAAError,
    InvalidNameError,
    evaluate_prepared,
    normalize_name,
    prepare_records,
)

MAX_CSR_BYTES = 64 * 1024
MAX_SAN_IDENTIFIERS = 20


class CSRError(CAAError):
    """Base class for CSR-level 422 failures (parse/signature/SAN)."""


def _normalize_identifier(raw: str) -> tuple[str, bool]:
    """Normalize one SAN DNS identifier to ``(lookup name, wildcard)``.

    Case and a single trailing dot are ignored. A wildcard is only the
    left-most ``*.`` prefix; it is stripped from the lookup name and
    reported through the flag. Any other ``*`` placement — or any name
    the base normalizer rejects — raises a 422-grade error.
    """
    name = raw.strip().lower()
    if name.endswith("."):
        name = name[:-1]
    wildcard = name.startswith("*.")
    if wildcard:
        name = name[2:]
    elif "*" in name:
        raise InvalidNameError(
            f"invalid identifier {raw!r}: wildcard only allowed as "
            "left-most '*.'"
        )
    return normalize_name(name, field="identifier"), wildcard


def parse_csr_identifiers(pem_text: str) -> tuple[str, list[tuple[str, bool]]]:
    """Verify a PEM CSR and return ``(sha256_of_der, identifiers)``.

    ``identifiers`` lists ``(lookup name, wildcard)`` pairs in SAN order.
    Every failure — oversized or unparseable PEM, an invalid
    self-signature, a missing subjectAltName extension, a non-DNS SAN
    entry, fewer than 1 or more than 20 identifiers, an illegal name, or
    identifiers that collide after normalization — raises a 422-grade
    :class:`CAAError`.
    """
    data = pem_text.encode("utf-8")
    if len(data) > MAX_CSR_BYTES:
        raise CSRError("CSR exceeds 64 KiB (65536 bytes)")
    try:
        csr = x509.load_pem_x509_csr(data)
    except ValueError as exc:
        raise CSRError(f"cannot parse PEM PKCS#10 CSR: {exc}") from exc
    try:
        signature_ok = csr.is_signature_valid
    except (ValueError, TypeError) as exc:  # unsupported key/signature type
        raise CSRError(f"cannot verify CSR signature: {exc}") from exc
    if not signature_ok:
        raise CSRError("CSR signature is invalid")
    try:
        san = csr.extensions.get_extension_for_oid(
            ExtensionOID.SUBJECT_ALTERNATIVE_NAME
        )
    except x509.ExtensionNotFound as exc:
        raise CSRError(
            "CSR carries no subjectAltName extension; the subject CN is "
            "not accepted as a fallback"
        ) from exc

    raw_names: list[str] = []
    for general_name in san.value:
        if not isinstance(general_name, x509.DNSName):
            raise CSRError(
                "subjectAltName contains a non-DNS identifier "
                f"({general_name.__class__.__name__})"
            )
        raw_names.append(general_name.value)
    if not 1 <= len(raw_names) <= MAX_SAN_IDENTIFIERS:
        raise CSRError(
            f"subjectAltName holds {len(raw_names)} DNS identifiers; "
            f"1-{MAX_SAN_IDENTIFIERS} required"
        )

    identifiers: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for raw in raw_names:
        name, wildcard = _normalize_identifier(raw)
        key = f"*.{name}" if wildcard else name
        if key in seen:
            raise CSRError(f"duplicate identifier {key!r} after normalization")
        seen.add(key)
        identifiers.append((name, wildcard))

    digest = hashlib.sha256(
        csr.public_bytes(serialization.Encoding.DER)
    ).hexdigest()
    return digest, identifiers


def check_csr(csr_pem: str, ca: str, records: list[dict]) -> dict:
    """Authorize every SAN identifier of a CSR against one record snapshot.

    The record set is prepared exactly once and shared by every
    identifier lookup; identifiers are evaluated in SAN order with the
    same rules as :func:`app.engine.evaluate`. Returns a dict with
    ``csr_sha256`` (SHA-256 of the CSR DER), ``accepted`` (true only
    when every identifier is authorized) and one entry per identifier
    holding its normalized ``name``, ``wildcard`` flag, ``authorized``
    verdict, ``effective_owner`` and lookup ``trace``.
    """
    digest, identifiers = parse_csr_identifiers(csr_pem)
    ca_name = normalize_name(ca, field="ca")
    snapshot = prepare_records(records)

    results = []
    for name, wildcard in identifiers:
        verdict = evaluate_prepared(name, wildcard, ca_name, snapshot)
        results.append(
            {
                "name": name,
                "wildcard": wildcard,
                "authorized": verdict["authorized"],
                "effective_owner": verdict["effective_owner"],
                "trace": verdict["trace"],
            }
        )
    return {
        "csr_sha256": digest,
        "accepted": all(r["authorized"] for r in results),
        "identifiers": results,
    }
