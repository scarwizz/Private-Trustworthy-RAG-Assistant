# src/security/encryptor.py
"""
Optional local encryption utilities using Fernet (symmetric).
Keeps key in a local file (key_path). Use with care: key must be backed up if you want to decrypt later.
"""
from cryptography.fernet import Fernet
from pathlib import Path
from typing import Dict, Any

class Encryptor:
    def __init__(self, key_path: str = "./.local_key.key"):
        self.key_path = Path(key_path)
        self.key = self._load_or_create_key()
        self.cipher = Fernet(self.key)

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            return self.key_path.read_bytes()
        else:
            k = Fernet.generate_key()
            self.key_path.write_bytes(k)
            # Restrict permissions if possible (best effort)
            try:
                import os, stat
                os.chmod(self.key_path, stat.S_IRUSR | stat.S_IWUSR)
            except Exception:
                pass
            return k

    def encrypt_str(self, plaintext: str) -> bytes:
        return self.cipher.encrypt(plaintext.encode("utf-8"))

    def decrypt_str(self, token: bytes) -> str:
        return self.cipher.decrypt(token).decode("utf-8")

def maybe_encrypt_metadata(metadata: Dict[str, Any], encryptor: Encryptor = None) -> Dict[str, Any]:
    """
    If encryptor is provided, encrypt all string values in metadata.
    Non-strings are left unchanged (but could be converted to str if desired).
    """
    if encryptor is None:
        return metadata
    out = {}
    for k, v in metadata.items():
        if isinstance(v, str):
            out[k] = encryptor.encrypt_str(v)
        else:
            out[k] = v
    return out
