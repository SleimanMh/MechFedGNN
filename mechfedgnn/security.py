"""Transport security: mutual TLS with identity bound to the certificate.

Uses established libraries only - `ssl` (OpenSSL) and `cryptography`. No custom
cryptography is implemented anywhere in this project.

DEVELOPMENT CERTIFICATES ONLY. `make_dev_certs` mints a local CA and short-lived
leaf certificates for 127.0.0.1. They are marked in their Common Name and
Organization and are NOT suitable for production or clinical deployment: the CA
key is written unencrypted next to the certs so tests can run unattended.

Identity rule: the authenticated client identity is the Common Name of the peer
certificate. A client id in a message body is never trusted on its own, and the
allowlist binds each certificate identity to the experiments it may join.
"""
import datetime as _dt
import ipaddress
import os
import ssl
from dataclasses import dataclass, field
from typing import Mapping, Sequence

DEV_ORG = "MechFedGNN-DEV-DO-NOT-USE-IN-PRODUCTION"


def make_dev_certs(out_dir: str, clients: Sequence[str], server_cn: str = "127.0.0.1",
                   days: int = 7) -> dict:
    """Mint a development CA, a server certificate and one per client.

    Returns the paths. Files are written 0600 where the platform supports it.
    """
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    os.makedirs(out_dir, exist_ok=True)
    now = _dt.datetime.now(_dt.timezone.utc)

    def _name(cn):
        return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn),
                          x509.NameAttribute(NameOID.ORGANIZATION_NAME, DEV_ORG)])

    def _write(path, data, secret=False):
        with open(path, "wb") as f:
            f.write(data)
        if secret:
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        return path

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_cert = (x509.CertificateBuilder()
               .subject_name(_name("MechFedGNN dev CA")).issuer_name(_name("MechFedGNN dev CA"))
               .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
               .not_valid_before(now - _dt.timedelta(minutes=5))
               .not_valid_after(now + _dt.timedelta(days=days))
               .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
               .sign(ca_key, hashes.SHA256()))
    paths = {"ca_cert": _write(os.path.join(out_dir, "ca.pem"),
                               ca_cert.public_bytes(serialization.Encoding.PEM)),
             "ca_key": _write(os.path.join(out_dir, "ca.key"),
                              ca_key.private_bytes(serialization.Encoding.PEM,
                                                   serialization.PrivateFormat.PKCS8,
                                                   serialization.NoEncryption()), secret=True)}

    def leaf(cn, filename, server=False):
        key = ec.generate_private_key(ec.SECP256R1())
        b = (x509.CertificateBuilder()
             .subject_name(_name(cn)).issuer_name(ca_cert.subject)
             .public_key(key.public_key()).serial_number(x509.random_serial_number())
             .not_valid_before(now - _dt.timedelta(minutes=5))
             .not_valid_after(now + _dt.timedelta(days=days))
             .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
             .add_extension(x509.ExtendedKeyUsage(
                 [x509.oid.ExtendedKeyUsageOID.SERVER_AUTH if server
                  else x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False))
        if server:
            b = b.add_extension(x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address(cn))] if _is_ip(cn)
                else [x509.DNSName(cn)]), critical=False)
        cert = b.sign(ca_key, hashes.SHA256())
        _write(os.path.join(out_dir, f"{filename}.pem"),
               cert.public_bytes(serialization.Encoding.PEM))
        _write(os.path.join(out_dir, f"{filename}.key"),
               key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                 serialization.NoEncryption()), secret=True)
        return os.path.join(out_dir, f"{filename}.pem")

    paths["server_cert"] = leaf(server_cn, "server", server=True)
    for c in clients:
        paths[f"client_{c}"] = leaf(c, f"client_{c}")
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write("DEVELOPMENT certificates for the MechFedGNN demo.\n"
                "Self-signed local CA, unencrypted keys, short lifetime.\n"
                "NOT suitable for production or clinical deployment.\n")
    return paths


def _is_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def server_context(cert_dir: str) -> ssl.SSLContext:
    """TLS with MUTUAL authentication: a client certificate is required and must
    chain to our CA."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(os.path.join(cert_dir, "server.pem"), os.path.join(cert_dir, "server.key"))
    ctx.load_verify_locations(os.path.join(cert_dir, "ca.pem"))
    ctx.verify_mode = ssl.CERT_REQUIRED          # mutual TLS
    return ctx


def client_context(cert_dir: str, client_id: str) -> ssl.SSLContext:
    """Verifies the SERVER certificate (hostname + chain) and presents this
    client's own certificate."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_verify_locations(os.path.join(cert_dir, "ca.pem"))
    ctx.load_cert_chain(os.path.join(cert_dir, f"client_{client_id}.pem"),
                        os.path.join(cert_dir, f"client_{client_id}.key"))
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


@dataclass
class Allowlist:
    """Binds a certificate identity to the experiments it may join.

    Authentication (who are you) comes from the certificate; authorisation
    (may you join THIS experiment) comes from here.
    """
    allowed: Mapping[str, Sequence[str]] = field(default_factory=dict)

    def check(self, identity: str, experiment_id: str) -> None:
        if identity not in self.allowed:
            raise PermissionError(f"unknown client certificate identity: {identity!r}")
        if experiment_id not in self.allowed[identity]:
            raise PermissionError(f"{identity!r} is not enrolled in {experiment_id!r}")
