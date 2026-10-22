"""Offline tests for the backup wizard's checks and credential save."""
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import credentials
import wizard_backup as B


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class BackupWizardTests(unittest.TestCase):
    def test_local_checks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.object(B.shutil, "which", return_value=None):
                checks = B.check_local(root)
            self.assertFalse(checks[0][1])
            self.assertFalse(checks[1][1])
            (root / "backup.py").write_text("")
            with patch.object(B.shutil, "which", return_value="/usr/bin/git"):
                checks = B.check_local(root)
            self.assertTrue(all(ok for _, ok, _ in checks))

    def test_github_private_and_missing(self):
        calls = []
        def private(req, timeout):
            calls.append((req.full_url, req.get_header("Authorization")))
            payload = ({"login": "alice"} if req.full_url.endswith("/user")
                       else {"private": True})
            return Response(json.dumps(payload).encode())
        info = B.check_github(" secret ", "my-backup", opener=private)
        self.assertEqual((info.owner, info.repo, info.exists, info.private),
                         ("alice", "my-backup", True, True))
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(auth == "Bearer secret" for _, auth in calls))

        def missing(req, timeout):
            if req.full_url.endswith("/user"):
                return Response(b'{"login":"alice"}')
            raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, None)
        info = B.check_github("secret", "my-backup", opener=missing)
        self.assertFalse(info.exists)
        self.assertIn("PRIVATE", info.detail)

    def test_reject_public_and_bad_token_without_leak(self):
        def public(req, timeout):
            payload = ({"login": "alice"} if req.full_url.endswith("/user")
                       else {"private": False})
            return Response(json.dumps(payload).encode())
        with self.assertRaisesRegex(B.BackupSetupError, "public"):
            B.check_github("SECRET_SENTINEL", "backup", opener=public)
        def forbidden(req, timeout):
            raise urllib.error.HTTPError(req.full_url, 403, "SECRET_SENTINEL", {}, None)
        with self.assertRaises(B.BackupSetupError) as caught:
            B.check_github("SECRET_SENTINEL", "backup", opener=forbidden)
        self.assertNotIn("SECRET_SENTINEL", str(caught.exception))
        with self.assertRaises(B.BackupSetupError):
            B.check_github("has spaces", "backup", opener=public)
        with self.assertRaises(B.BackupSetupError):
            B.repo_name("../wrong")

    def test_save_private_env_and_config_merge(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env, config = root / ".env", root / "config.json"
            config.write_text('{"active_provider":"anthropic","backup":{"keep":3}}')
            with patch.object(credentials, "ENV_PATH", env):
                B.save(" tok$#'en ", " DomicileBackup ", config_path=config)
                self.assertEqual(credentials.get("GITHUB_TOKEN"), "tok$#'en")
                self.assertEqual(env.stat().st_mode & 0o777, 0o600)
                self.assertNotIn("tok$#'en", config.read_text())
                self.assertEqual(json.loads(config.read_text()),
                                 {"active_provider": "anthropic",
                                  "backup": {"keep": 3, "repo": "DomicileBackup"}})

    def test_bad_config_does_not_write_secret(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env, config = root / ".env", root / "config.json"
            config.write_text("{broken")
            with patch.object(credentials, "ENV_PATH", env):
                with self.assertRaises(B.BackupSetupError):
                    B.save("secret", "backup", config_path=config)
            self.assertFalse(env.exists())


if __name__ == "__main__":
    unittest.main()
