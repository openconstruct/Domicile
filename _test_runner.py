"""Run one test script with disposable data paths and provider fixtures."""
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import credentials
from unittest.mock import patch

SANDBOX_ENV = "DOMICILE_TEST_SANDBOX"


def require_sandbox() -> None:
    """Abort unless this test is running under _test_runner's disposable root.

    Tests that mutate rollover marks or append to rollover.log must never be
    invoked as `python3 _foo_test.py` -- Path.home() is unpatched there, so
    paths.py resolves to the live data dir and the test writes real provider
    marks (for the seconds between mark and clear, a timer-driven turn can
    roll over for no reason) and forges incident lines in rollover.log. That
    happened on 2026-09-24 22:00:56 and poisoned the auth-expiry evidence.
    """
    import paths
    root = os.environ.get(SANDBOX_ENV)
    if not root or paths.HOME != Path(root) or Path(root).resolve() not in paths.CCHAT_DATA.parents:
        sys.exit(
            "refusing to run outside the test sandbox: this test mutates "
            "provider-rollover state.\n  run it as: python3 _test_runner.py "
            + Path(sys.argv[0]).name)


def main():
    target = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="domicile-test-") as directory:
        root = Path(directory)
        data = root / "domicile" / "data"
        logs = data / "mem0" / "logs"
        logs.mkdir(parents=True)
        (logs / "2026-09-01.log").write_text(
            "Compaction ceiling and COMPACT_BASE_PREAMBLE. qdrant, core-memories.md, ddgs.\n")
        env_file = root / ".env"
        env_file.write_text("FIXTURE_API_KEY=test-only-not-a-real-key\n")
        providers = {"anthropic": {"adapter": "claude_cli", "models": {
            "chat": ["sonnet"], "code": ["opus"], "compact": ["haiku"]}}}
        for name in ("openrouter", "ali"):
            providers[name] = {"adapter": "openai_compatible", "key_env": "FIXTURE_API_KEY",
                               "base_url": "http://127.0.0.1:1/v1", "models": {
                                   "chat": ["test"], "code": ["test"], "compact": ["test"]}}
        (data / "config.json").write_text(json.dumps({
            "active_provider": "anthropic", "provider_order": list(providers),
            "providers": providers}))
        # paths.py reads DOMICILE_DATA and Path.home() when first imported, so both
        # must be in place before that import -- never import paths above this line.
        assert "paths" not in sys.modules, "paths imported before the sandbox was set up"
        with patch.dict(os.environ, {SANDBOX_ENV: str(root), "DOMICILE_DATA": str(data)}), \
                patch.object(Path, "home", return_value=root):
            import paths
            if root.resolve() not in paths.CCHAT_DATA.parents:
                sys.exit(f"sandbox setup failed: data dir is {paths.CCHAT_DATA}, not under {root}")
            with patch.object(credentials, "ENV_PATH", env_file), \
                    patch.object(paths, "ROOT", root / "domicile"), \
                    patch.object(tempfile, "tempdir", directory), \
                    patch.object(sys, "argv", [str(target)]):
                runpy.run_path(str(target), run_name="__main__")

if __name__ == "__main__":
    main()
