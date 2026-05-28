"""Generate self-signed TLS certificates for Piper AI Agent inter-service communication."""

import ipaddress
import os
import datetime
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

CERTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "certs")

# All Docker service hostnames + localhost for local dev
SERVICE_HOSTNAMES = [
    "localhost",
    "gateway_server",
    "agent_service",
    "memory_service",
    "llm_service",
    "knowledge_service",
    "tool_service",
    "recommendation_service",
]


def generate_ca():
    """Generate a self-signed CA certificate and private key."""
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)

    ca_name = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Piper AI Dev CA"),
        x509.NameAttribute(NameOID.COMMON_NAME, "Piper Dev Root CA"),
    ])

    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    return ca_key, ca_cert


def generate_server_cert(ca_key, ca_cert):
    """Generate a server certificate signed by the CA with SANs for all services."""
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    server_name = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Piper AI"),
        x509.NameAttribute(NameOID.COMMON_NAME, "piper-services"),
    ])

    san_entries = [x509.DNSName(h) for h in SERVICE_HOSTNAMES]
    san_entries.append(x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")))
    san_entries.append(x509.IPAddress(ipaddress.IPv6Address("::1")))

    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(san_entries),
            critical=False,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    return server_key, server_cert


def write_pem(path, data):
    with open(path, "wb") as f:
        f.write(data)
    print(f"  Written: {path}")


def main():
    os.makedirs(CERTS_DIR, exist_ok=True)
    print(f"Generating certificates in {CERTS_DIR} ...")

    ca_key, ca_cert = generate_ca()

    write_pem(
        os.path.join(CERTS_DIR, "ca-key.pem"),
        ca_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ),
    )
    write_pem(
        os.path.join(CERTS_DIR, "ca.pem"),
        ca_cert.public_bytes(serialization.Encoding.PEM),
    )

    server_key, server_cert = generate_server_cert(ca_key, ca_cert)

    write_pem(
        os.path.join(CERTS_DIR, "server-key.pem"),
        server_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ),
    )
    write_pem(
        os.path.join(CERTS_DIR, "server.pem"),
        server_cert.public_bytes(serialization.Encoding.PEM),
    )

    print("\nDone. Files generated:")
    for f in ["ca.pem", "ca-key.pem", "server.pem", "server-key.pem"]:
        print(f"  certs/{f}")


if __name__ == "__main__":
    main()
