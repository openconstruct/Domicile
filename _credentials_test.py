"""Credential consumers exercised with a private, disposable .env."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import credentials
import config_loader
import backup
import discord_bridge
from mem0_llm_common import read_key
from providers.openai_compatible import _read_key


def main():
    with tempfile.TemporaryDirectory() as directory:
        env = Path(directory) / '.env'
        legacy = Path(directory) / 'fixture.key'
        legacy.write_text('obsolete')
        with patch.object(credentials, 'ENV_PATH', env):
            credentials.save({'FIXTURE_API_KEY': "literal$VALUE#with'quote",
                              'GITHUB_TOKEN': 'fixture-github',
                              'GITLAB_TOKEN': 'fixture-gitlab',
                              'CODEBERG_TOKEN': 'fixture-codeberg'})  # only GitHub is a backup target
            assert env.stat().st_mode & 0o777 == 0o600
            with patch.dict(os.environ, {'FIXTURE_API_KEY': 'stale-process-value'}):
                assert read_key(str(legacy)) == "literal$VALUE#with'quote"
                assert _read_key(str(legacy), 'FIXTURE_API_KEY') == read_key(str(legacy))
            cfg = {'providers': {'fixture': {'adapter': 'openai_compatible',
                                             'key_file': str(legacy)}}}
            assert config_loader.key_available('fixture', cfg)
            discord_bridge.save_config('fake-bot', 123, 456, True)
            assert discord_bridge.load_config() == ('fake-bot', 123, 456, True)
            assert credentials.get('FIXTURE_API_KEY') == "literal$VALUE#with'quote"
            discord_bridge.save_config('replacement', 123)
            assert discord_bridge.load_config() == ('replacement', 123, None, False)
            assert backup.read_tokens() == {'github': 'fixture-github'}
            credentials.save({'FIXTURE_API_KEY': ''})
            assert not config_loader.key_available('fixture', cfg)
            try:
                read_key(str(legacy))
            except RuntimeError:
                pass
            else:
                raise AssertionError('Legacy file unexpectedly used')
            env.unlink()
            assert credentials.get('FIXTURE_API_KEY') == ''
    print('PASS: .env round trips, precedence, permissions, providers, mem0, Discord, backup, missing credentials')


if __name__ == '__main__':
    main()
