import sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wizard_service as S

class T(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp()); self.i = self.d/"cchat"; self.i.mkdir()
        (self.i/"cchat.py").write_text(""); (self.i/"watchdog.py").write_text("")
        self.u = self.d/"units"
    def test_render_paths(self):
        u = S.render(self.i, python="/usr/bin/python3", tmux="/usr/bin/tmux")
        self.assertIn(f"{self.i.resolve()}/cchat.py", u["cchat.service"])
        self.assertIn("RestartPreventExitStatus=3", u["cchat.service"])
        self.assertEqual(len(u), 3)
    def test_install_create_only(self):
        S.install(self.i, self.u, python="/usr/bin/python3", tmux="/usr/bin/tmux")
        self.assertEqual(len(list(self.u.iterdir())), 3)
        (self.u/"cchat.service").write_text("mine")
        with self.assertRaises(S.ServiceError): S.install(self.i, self.u, tmux="/usr/bin/tmux")
        self.assertEqual((self.u/"cchat.service").read_text(), "mine")
    def test_wrong_dir(self):
        with self.assertRaises(S.ServiceError): S.render(self.d, tmux="/usr/bin/tmux")
    def test_gmail_units_only_when_asked_and_exit3_is_success(self):
        kw = dict(python="/usr/bin/python3", tmux="/usr/bin/tmux")
        self.assertNotIn("domicile-gmail.timer", S.render(self.i, **kw))
        with self.assertRaises(S.ServiceError): S.render(self.i, gmail=True, **kw)   # no gmail_notify.py
        (self.i/"google_auth").mkdir(); (self.i/"google_auth"/"gmail_notify.py").write_text("")
        u = S.render(self.i, gmail=True, **kw)
        self.assertEqual(len(u), 5)
        self.assertIn(f"{self.i.resolve()}/google_auth/gmail_notify.py", u["domicile-gmail.service"])
        self.assertIn("SuccessExitStatus=3", u["domicile-gmail.service"])
        self.assertIn("OnUnitActiveSec=5min", u["domicile-gmail.timer"])
        S.install(self.i, self.u, gmail=True, **kw)
        self.assertEqual(len(list(self.u.iterdir())), 5)
    def test_data_dir_override_reaches_every_unit_or_none(self):
        import os
        from unittest.mock import patch
        kw = dict(python="/usr/bin/python3", tmux="/usr/bin/tmux")
        (self.i/"google_auth").mkdir(exist_ok=True); (self.i/"google_auth"/"gmail_notify.py").write_text("")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DOMICILE_DATA", None)
            plain = S.render(self.i, gmail=True, **kw)
        self.assertFalse(any("DOMICILE_DATA" in t for t in plain.values()))
        with patch.dict(os.environ, {"DOMICILE_DATA": "/srv/agent-data"}):
            custom = S.render(self.i, gmail=True, **kw)
        for name in ("cchat.service", "cchat-watchdog.service", "domicile-gmail.service"):
            self.assertIn("Environment=DOMICILE_DATA=", custom[name], name)
        self.assertNotIn("DOMICILE_DATA", custom["cchat-watchdog.timer"])
    def test_rss_units_only_when_asked_and_carry_the_data_dir(self):
        import os
        from unittest.mock import patch
        kw = dict(python="/usr/bin/python3", tmux="/usr/bin/tmux")
        self.assertNotIn("domicile-rss.timer", S.render(self.i, **kw))
        with self.assertRaises(S.ServiceError): S.render(self.i, rss=True, **kw)   # no rss_notify.py
        (self.i/"rss").mkdir(); (self.i/"rss"/"rss_notify.py").write_text("")
        with patch.dict(os.environ, {"DOMICILE_DATA": "/srv/d"}):
            u = S.render(self.i, rss=True, **kw)
        self.assertEqual(len(u), 5)
        self.assertIn(f"{self.i.resolve()}/rss/rss_notify.py", u["domicile-rss.service"])
        import paths
        self.assertIn(f"Environment=DOMICILE_DATA={paths.CCHAT_DATA}\n", u["domicile-rss.service"])
        self.assertIn("OnUnitActiveSec=1h", u["domicile-rss.timer"])
if __name__ == "__main__": unittest.main()
