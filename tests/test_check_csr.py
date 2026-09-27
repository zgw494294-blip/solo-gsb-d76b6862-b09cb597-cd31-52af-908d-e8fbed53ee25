"""End-to-end tests for POST /v1/issuance/check-csr."""

import hashlib
import ipaddress

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from fastapi.testclient import TestClient

import app.csr as csr_mod
from app.main import app

client = TestClient(app)

_KEY = ec.generate_private_key(ec.SECP256R1())


def make_csr(names=(), san_extra=None, with_san=True):
    """Build a signed CSR carrying the given SAN DNS names (and extras)."""
    builder = x509.CertificateSigningRequestBuilder().subject_name(
        x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "cn.example.com")])
    )
    if with_san:
        general_names = [x509.DNSName(n) for n in names]
        if san_extra:
            general_names.extend(san_extra)
        builder = builder.add_extension(
            x509.SubjectAlternativeName(general_names), critical=False
        )
    return builder.sign(_KEY, hashes.SHA256())


def pem_of(csr):
    return csr.public_bytes(serialization.Encoding.PEM).decode()


def der_of(csr):
    return csr.public_bytes(serialization.Encoding.DER)


def caa(owner, tag, value="", flag=0):
    return {"type": "CAA", "owner": owner, "flag": flag, "tag": tag, "value": value}


def cname(owner, target):
    return {"type": "CNAME", "owner": owner, "target": target}


def post(body):
    return client.post("/v1/issuance/check-csr", json=body)


def body(names, ca="letsencrypt.org", records=None, **kwargs):
    return {
        "csr": pem_of(make_csr(names, **kwargs)),
        "ca": ca,
        "records": records or [],
    }


LE_AT_EXAMPLE = [caa("example.com", "issue", "letsencrypt.org")]


# ---------- happy paths ----------

def test_single_identifier_authorized():
    csr = make_csr(["www.example.com"])
    r = post(
        {
            "csr": pem_of(csr),
            "ca": "letsencrypt.org",
            "records": LE_AT_EXAMPLE,
        }
    )
    assert r.status_code == 200
    data = r.json()
    assert data["csr_sha256"] == hashlib.sha256(der_of(csr)).hexdigest()
    assert data["accepted"] is True
    assert len(data["identifiers"]) == 1
    ident = data["identifiers"][0]
    assert ident["name"] == "www.example.com"
    assert ident["wildcard"] is False
    assert ident["authorized"] is True
    assert ident["effective_owner"] == "example.com"
    # trace walked www.example.com (miss) then example.com (hit)
    assert [s["query"] for s in ident["trace"]] == [
        "www.example.com",
        "example.com",
    ]


def test_multiple_identifiers_keep_san_order():
    r = post(
        body(
            ["b.example.com", "a.example.com", "c.example.com"],
            records=LE_AT_EXAMPLE,
        )
    )
    assert r.status_code == 200
    data = r.json()
    assert data["accepted"] is True
    assert [i["name"] for i in data["identifiers"]] == [
        "b.example.com",
        "a.example.com",
        "c.example.com",
    ]


def test_any_unauthorized_identifier_rejects_whole_csr():
    r = post(
        body(
            ["allowed.example.com", "denied.other.net", "also.example.com"],
            records=LE_AT_EXAMPLE + [caa("other.net", "issue", "pki.goog")],
        )
    )
    assert r.status_code == 200
    data = r.json()
    assert data["accepted"] is False
    # every identifier is still evaluated and reported, in SAN order
    assert [
        (i["name"], i["authorized"], i["effective_owner"])
        for i in data["identifiers"]
    ] == [
        ("allowed.example.com", True, "example.com"),
        ("denied.other.net", False, "other.net"),
        ("also.example.com", True, "example.com"),
    ]


def test_wildcard_identifier_strips_prefix_and_uses_issuewild():
    # issue allows LE but issuewild explicitly denies everyone, so the
    # wildcard identifier must not be authorized.
    r = post(
        body(
            ["*.example.com"],
            records=[
                caa("example.com", "issue", "letsencrypt.org"),
                caa("example.com", "issuewild", ""),
            ],
        )
    )
    assert r.status_code == 200
    data = r.json()
    assert data["accepted"] is False
    ident = data["identifiers"][0]
    assert ident["name"] == "example.com"
    assert ident["wildcard"] is True
    assert ident["authorized"] is False


def test_identifier_case_and_trailing_dot_normalized():
    r = post(body(["WWW.Example.COM."], records=LE_AT_EXAMPLE))
    assert r.status_code == 200
    ident = r.json()["identifiers"][0]
    assert ident["name"] == "www.example.com"
    assert ident["authorized"] is True


def test_wildcard_and_base_name_are_distinct_identifiers():
    r = post(body(["*.example.com", "example.com"], records=LE_AT_EXAMPLE))
    assert r.status_code == 200
    data = r.json()
    assert data["accepted"] is True
    assert [(i["name"], i["wildcard"]) for i in data["identifiers"]] == [
        ("example.com", True),
        ("example.com", False),
    ]


def test_no_caa_records_authorizes_everyone():
    r = post(body(["a.example.com", "b.example.com"]))
    data = r.json()
    assert data["accepted"] is True
    assert all(i["effective_owner"] is None for i in data["identifiers"])


def test_cname_alias_followed_per_identifier():
    r = post(
        body(
            ["www.example.com"],
            records=[
                cname("www.example.com", "edge.cdn.example"),
                caa("edge.cdn.example", "issue", "letsencrypt.org"),
            ],
        )
    )
    ident = r.json()["identifiers"][0]
    assert ident["authorized"] is True
    assert ident["effective_owner"] == "edge.cdn.example"
    assert ident["trace"][0]["kind"] == "cname"
    assert ident["trace"][0]["cname_target"] == "edge.cdn.example"


def test_twenty_identifiers_accepted():
    names = [f"h{i:02d}.example.com" for i in range(20)]
    r = post(body(names, records=LE_AT_EXAMPLE))
    assert r.status_code == 200
    assert len(r.json()["identifiers"]) == 20


def test_ca_name_normalized():
    r = post(body(["www.example.com"], ca="LetsEncrypt.ORG.", records=LE_AT_EXAMPLE))
    assert r.status_code == 200
    assert r.json()["accepted"] is True


def test_records_prepared_once_for_all_identifiers(monkeypatch):
    calls = []
    real_prepare = csr_mod.prepare_records

    def spy(records):
        calls.append(list(records))
        return real_prepare(records)

    monkeypatch.setattr(csr_mod, "prepare_records", spy)
    r = post(
        body(
            ["a.example.com", "b.example.com", "c.example.com"],
            records=LE_AT_EXAMPLE,
        )
    )
    assert r.status_code == 200
    assert len(calls) == 1


# ---------- 422: CSR structure ----------

def test_missing_san_is_422_and_cn_is_not_a_fallback():
    # CN alone would match the record set, but only SAN counts.
    csr = make_csr(with_san=False)
    r = post({"csr": pem_of(csr), "ca": "letsencrypt.org", "records": LE_AT_EXAMPLE})
    assert r.status_code == 422
    assert r.json()["error_type"] == "CSRError"


def test_non_dns_san_entries_are_422():
    for extra in (
        [x509.IPAddress(ipaddress.ip_address("10.0.0.1"))],
        [x509.RFC822Name("admin@example.com")],
        [x509.UniformResourceIdentifier("https://example.com/")],
    ):
        r = post(body(["www.example.com"], san_extra=extra, records=LE_AT_EXAMPLE))
        assert r.status_code == 422, extra
        assert r.json()["error_type"] == "CSRError"


def test_empty_san_is_422():
    r = post(body([], records=LE_AT_EXAMPLE))
    assert r.status_code == 422
    assert r.json()["error_type"] == "CSRError"


def test_twenty_one_identifiers_is_422():
    names = [f"h{i:02d}.example.com" for i in range(21)]
    r = post(body(names, records=LE_AT_EXAMPLE))
    assert r.status_code == 422
    assert r.json()["error_type"] == "CSRError"


def test_duplicate_names_after_normalization_is_422():
    r = post(body(["WWW.Example.com", "www.example.COM."], records=LE_AT_EXAMPLE))
    assert r.status_code == 422
    assert r.json()["error_type"] == "CSRError"


def test_duplicate_wildcards_after_normalization_is_422():
    r = post(body(["*.Example.com", "*.example.COM."], records=LE_AT_EXAMPLE))
    assert r.status_code == 422
    assert r.json()["error_type"] == "CSRError"


# ---------- 422: identifier names ----------

@pytest.mark.parametrize(
    "name",
    [
        "foo.*.example.com",  # wildcard not left-most
        "*foo.example.com",  # partial-label wildcard
        "*",  # lone star
        "bad..com",  # empty label
        "foo_bar.example.com",  # illegal character
        "a" * 64 + ".com",  # label too long
        "-x.com",  # leading hyphen
    ],
)
def test_illegal_identifier_names_are_422(name):
    r = post(body([name], records=LE_AT_EXAMPLE))
    assert r.status_code == 422, name
    assert r.json()["error_type"] == "InvalidNameError"


def test_star_dot_only_is_422():
    # "*." strips to the empty name.
    r = post(body(["*."], records=LE_AT_EXAMPLE))
    assert r.status_code == 422
    assert r.json()["error_type"] == "InvalidNameError"


# ---------- 422: PEM / signature ----------

def test_tampered_signature_is_422():
    der = bytearray(der_of(make_csr(["www.example.com"])))
    der[-1] ^= 0x01  # flip one bit inside the signature
    pem = pem_of(x509.load_der_x509_csr(bytes(der)))
    r = post({"csr": pem, "ca": "letsencrypt.org", "records": LE_AT_EXAMPLE})
    assert r.status_code == 422
    assert r.json()["error_type"] == "CSRError"
    assert "signature" in r.json()["detail"]


def test_garbage_pem_is_422():
    r = post({"csr": "not a pem at all", "ca": "letsencrypt.org", "records": []})
    assert r.status_code == 422
    assert r.json()["error_type"] == "CSRError"


def test_non_csr_pem_is_422():
    pem = _KEY.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    r = post({"csr": pem, "ca": "letsencrypt.org", "records": []})
    assert r.status_code == 422
    assert r.json()["error_type"] == "CSRError"


def test_oversized_csr_is_422():
    r = post({"csr": "A" * 65537, "ca": "letsencrypt.org", "records": []})
    assert r.status_code == 422
    assert r.json()["error_type"] == "CSRError"
    assert "64 KiB" in r.json()["detail"]


def test_empty_csr_string_is_422():
    r = post({"csr": "", "ca": "letsencrypt.org", "records": []})
    assert r.status_code == 422


# ---------- 422: records / aliases / CA ----------

def test_cname_loop_is_422():
    r = post(
        body(
            ["x.test"],
            records=[cname("x.test", "y.test"), cname("y.test", "x.test")],
        )
    )
    assert r.status_code == 422
    assert r.json()["error_type"] == "CNAMELoopError"


def test_alias_error_on_later_identifier_is_422():
    # The first identifier resolves cleanly; the loop sits on the second.
    r = post(
        body(
            ["www.example.com", "x.test"],
            records=LE_AT_EXAMPLE
            + [cname("x.test", "y.test"), cname("y.test", "x.test")],
        )
    )
    assert r.status_code == 422
    assert r.json()["error_type"] == "CNAMELoopError"


def test_mixed_records_is_422():
    r = post(
        body(
            ["x.test"],
            records=[cname("x.test", "z.test"), caa("x.test", "issue", "a.net")],
        )
    )
    assert r.status_code == 422
    assert r.json()["error_type"] == "MixedRecordsError"


def test_multiple_cname_is_422():
    r = post(
        body(
            ["x.test"],
            records=[cname("x.test", "a.test"), cname("x.test", "b.test")],
        )
    )
    assert r.status_code == 422
    assert r.json()["error_type"] == "MultipleCNAMEError"


def test_more_than_100_records_is_422():
    recs = [caa(f"h{i:03d}.example.com", "issue", "a.net") for i in range(101)]
    r = post(body(["www.example.com"], records=recs))
    assert r.status_code == 422


def test_bad_ca_name_is_422():
    r = post(body(["www.example.com"], ca="bad..com", records=LE_AT_EXAMPLE))
    assert r.status_code == 422
