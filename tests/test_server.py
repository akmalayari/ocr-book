"""
test_server.py — Tests unitaires rigoureux pour server.py

Stratégie de mock :
  - subprocess.Popen est TOUJOURS mocké : on ne lance jamais un vrai processus.
  - requests.get est mocké pour simuler les réponses du serveur.
  - Les tests du context manager vérifient que stop_server est appelé dans tous
    les cas (succès, exception dans le bloc with, Ctrl+C).

Couvre :
  - _wait_for_server : succès immédiat, succès après délai, timeout, erreur connexion
  - start_server     : succès, timeout, commande construite correctement, verbose
  - stop_server      : processus vivant, processus déjà mort, timeout kill
  - nexa_server      : entrée/sortie normales, exception dans le bloc, restauration SIGINT
  - NexaServerError  : hiérarchie d'héritage
"""

import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from server import NexaServerError, _wait_for_server, nexa_server, start_server, stop_server


# ── Fixture de config ─────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return Config(
        model="NexaAI/TestModel",
        port=19999,
        server_timeout_s=3,
        verbose=False,
        log_file="",
    )


@pytest.fixture
def cfg_verbose():
    return Config(
        model="NexaAI/TestModel",
        port=19999,
        server_timeout_s=3,
        verbose=True,
        log_file="",
    )


def make_proc(poll_return=None):
    """Crée un mock Popen. poll_return=None → processus vivant."""
    proc = MagicMock(spec=subprocess.Popen)
    proc.poll.return_value = poll_return
    proc.pid = 12345
    return proc


# ── NexaServerError ───────────────────────────────────────────────────────────

class TestNexaServerError:

    def test_is_runtime_error_subclass(self):
        assert issubclass(NexaServerError, RuntimeError)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(NexaServerError, match="test"):
            raise NexaServerError("test")

    def test_message_preserved(self):
        err = NexaServerError("message d'erreur")
        assert "message d'erreur" in str(err)


# ── _wait_for_server ──────────────────────────────────────────────────────────

class TestWaitForServer:

    def test_returns_true_on_200_immediately(self):
        proc = make_proc()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("server.requests.get", return_value=mock_resp) as mock_get:
            with patch("server.time.sleep"):
                result = _wait_for_server(proc, port=19999, timeout_s=5)
        assert result is True
        mock_get.assert_called_once()

    def test_returns_true_after_retry(self):
        """Simule : échec × 2 puis succès."""
        proc = make_proc()
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        side_effects = [
            requests.ConnectionError("not ready"),
            requests.ConnectionError("not ready"),
            ok_resp,
        ]
        with patch("server.requests.get", side_effect=side_effects):
            with patch("server.time.sleep"):
                with patch("server.time.time", side_effect=[0, 1, 2, 3, 100]):
                    result = _wait_for_server(proc, port=19999, timeout_s=10)
        assert result is True

    def test_returns_false_on_timeout(self):
        proc = make_proc()
        with patch("server.requests.get", side_effect=requests.ConnectionError):
            with patch("server.time.sleep"):
                time_values = iter([0] + [i * 2 for i in range(1, 20)])
                with patch("server.time.time", side_effect=time_values):
                    result = _wait_for_server(proc, port=19999, timeout_s=3)
        assert result is False

    def test_returns_false_when_status_not_200(self):
        """Un serveur qui répond 500 ne doit pas être considéré comme prêt."""
        proc = make_proc()
        bad_resp = MagicMock()
        bad_resp.status_code = 500
        time_values = iter([0] + [i * 2 for i in range(1, 20)])
        with patch("server.requests.get", return_value=bad_resp):
            with patch("server.time.sleep"):
                with patch("server.time.time", side_effect=time_values):
                    result = _wait_for_server(proc, port=19999, timeout_s=3)
        assert result is False

    def test_polls_correct_url(self):
        proc = make_proc()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("server.requests.get", return_value=mock_resp) as mock_get:
            with patch("server.time.sleep"):
                _wait_for_server(proc, port=12345, timeout_s=5)
        called_url = mock_get.call_args[0][0]
        assert "12345" in called_url
        assert "/v1/models" in called_url

    def test_sleeps_between_attempts(self):
        """Vérifie que time.sleep est appelé entre les tentatives."""
        proc = make_proc()
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        side_effects = [requests.ConnectionError(), ok_resp]
        with patch("server.requests.get", side_effect=side_effects):
            with patch("server.time.sleep") as mock_sleep:
                with patch("server.time.time", side_effect=[0, 1, 2, 100]):
                    _wait_for_server(proc, port=19999, timeout_s=10)
        mock_sleep.assert_called()

    def test_uses_short_timeout_for_get_requests(self):
        """Chaque GET doit avoir son propre timeout court (≤ 5s)."""
        proc = make_proc()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("server.requests.get", return_value=mock_resp) as mock_get:
            with patch("server.time.sleep"):
                _wait_for_server(proc, port=19999, timeout_s=5)
        _, kwargs = mock_get.call_args
        assert "timeout" in kwargs
        assert kwargs["timeout"] <= 5


# ── start_server ──────────────────────────────────────────────────────────────

class TestStartServer:

    def test_returns_proc_on_success(self, cfg):
        proc = make_proc()
        with patch("server.subprocess.Popen", return_value=proc):
            with patch("server._wait_for_server", return_value=True):
                result = start_server(cfg)
        assert result is proc

    def test_raises_nexaservererror_on_timeout(self, cfg):
        proc = make_proc()
        with patch("server.subprocess.Popen", return_value=proc):
            with patch("server._wait_for_server", return_value=False):
                with pytest.raises(NexaServerError):
                    start_server(cfg)

    def test_kills_proc_on_timeout(self, cfg):
        proc = make_proc()
        with patch("server.subprocess.Popen", return_value=proc):
            with patch("server._wait_for_server", return_value=False):
                with pytest.raises(NexaServerError):
                    start_server(cfg)
        proc.kill.assert_called_once()

    def test_error_message_contains_model_name(self, cfg):
        proc = make_proc()
        with patch("server.subprocess.Popen", return_value=proc):
            with patch("server._wait_for_server", return_value=False):
                with pytest.raises(NexaServerError, match=cfg.model):
                    start_server(cfg)

    def test_error_message_contains_timeout_value(self, cfg):
        proc = make_proc()
        with patch("server.subprocess.Popen", return_value=proc):
            with patch("server._wait_for_server", return_value=False):
                with pytest.raises(NexaServerError, match=str(cfg.server_timeout_s)):
                    start_server(cfg)

    def test_command_contains_nexa_serve(self, cfg):
        proc = make_proc()
        with patch("server.subprocess.Popen", return_value=proc) as mock_popen:
            with patch("server._wait_for_server", return_value=True):
                start_server(cfg)
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "nexa"
        assert "serve" in cmd


    def test_verbose_uses_pipe(self, cfg_verbose):
        proc = make_proc()
        with patch("server.subprocess.Popen", return_value=proc) as mock_popen:
            with patch("server._wait_for_server", return_value=True):
                start_server(cfg_verbose)
        kwargs = mock_popen.call_args[1]
        assert kwargs.get("stdout") == subprocess.PIPE
        assert kwargs.get("stderr") == subprocess.PIPE

    def test_non_verbose_uses_devnull(self, cfg):
        proc = make_proc()
        with patch("server.subprocess.Popen", return_value=proc) as mock_popen:
            with patch("server._wait_for_server", return_value=True):
                start_server(cfg)
        kwargs = mock_popen.call_args[1]
        assert kwargs.get("stdout") == subprocess.DEVNULL
        assert kwargs.get("stderr") == subprocess.DEVNULL


# ── stop_server ───────────────────────────────────────────────────────────────

class TestStopServer:

    def test_terminates_running_process(self):
        proc = make_proc(poll_return=None)
        with patch("server.sys.platform", "linux"):
            stop_server(proc)
        proc.terminate.assert_called_once()

    def test_waits_after_terminate(self):
        proc = make_proc(poll_return=None)
        with patch("server.sys.platform", "linux"):
            stop_server(proc)
        proc.wait.assert_called_once()

    def test_kills_if_wait_times_out(self):
        proc = make_proc(poll_return=None)
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="nexa", timeout=5)
        with patch("server.sys.platform", "linux"):
            stop_server(proc)
        proc.kill.assert_called_once()

    def test_does_nothing_if_process_already_dead(self):
        """poll() != None → processus déjà terminé → pas de terminate/kill."""
        proc = make_proc(poll_return=0)  # code retour 0 = processus terminé
        with patch("server.sys.platform", "linux"):
            stop_server(proc)
        proc.terminate.assert_not_called()
        proc.kill.assert_not_called()

    def test_does_nothing_if_proc_is_none(self):
        """Ne doit pas lever d'exception si proc est None."""
        stop_server(None)  # pas d'exception attendue

    def test_wait_timeout_is_5_seconds(self):
        proc = make_proc(poll_return=None)
        with patch("server.sys.platform", "linux"):
            stop_server(proc)
        _, kwargs = proc.wait.call_args
        assert kwargs.get("timeout") == 5


# ── nexa_server (context manager) ────────────────────────────────────────────

class TestNexaServerContextManager:

    def test_yields_proc(self, cfg):
        proc = make_proc()
        with patch("server.start_server", return_value=proc):
            with patch("server.stop_server") as mock_stop:
                with nexa_server(cfg) as yielded:
                    assert yielded is proc

    def test_stop_called_on_normal_exit(self, cfg):
        proc = make_proc()
        with patch("server.start_server", return_value=proc):
            with patch("server.stop_server") as mock_stop:
                with nexa_server(cfg):
                    pass
        mock_stop.assert_called_once_with(proc)

    def test_stop_called_on_exception_in_block(self, cfg):
        """stop_server doit être appelé même si une exception est levée dans le bloc."""
        proc = make_proc()
        with patch("server.start_server", return_value=proc):
            with patch("server.stop_server") as mock_stop:
                with pytest.raises(ValueError):
                    with nexa_server(cfg):
                        raise ValueError("erreur dans le bloc")
        mock_stop.assert_called_once_with(proc)

    def test_exception_propagates_after_stop(self, cfg):
        """L'exception levée dans le bloc doit remonter après l'arrêt du serveur."""
        proc = make_proc()
        with patch("server.start_server", return_value=proc):
            with patch("server.stop_server"):
                with pytest.raises(RuntimeError, match="propagée"):
                    with nexa_server(cfg):
                        raise RuntimeError("propagée")

    def test_sigint_handler_installed(self, cfg):
        """Vérifie que le handler SIGINT est installé pendant le bloc with."""
        proc = make_proc()
        captured_handler = {}

        def fake_signal(signum, handler):
            if signum == signal.SIGINT:
                captured_handler["fn"] = handler

        with patch("server.start_server", return_value=proc):
            with patch("server.stop_server"):
                with patch("server.signal.signal", side_effect=fake_signal):
                    with patch("server.signal.getsignal", return_value=signal.SIG_DFL):
                        with nexa_server(cfg):
                            pass
        # Le handler personnalisé doit avoir été installé
        assert "fn" in captured_handler

    def test_original_sigint_restored_after_exit(self, cfg):
        """Le handler SIGINT original doit être restauré à la sortie du bloc."""
        proc = make_proc()
        original_handler = signal.SIG_DFL
        installed_handlers = []

        def fake_signal(signum, handler):
            if signum == signal.SIGINT:
                installed_handlers.append(handler)

        with patch("server.start_server", return_value=proc):
            with patch("server.stop_server"):
                with patch("server.signal.signal", side_effect=fake_signal):
                    with patch("server.signal.getsignal", return_value=original_handler):
                        with nexa_server(cfg):
                            pass

        # Le dernier handler installé doit être le handler original
        assert installed_handlers[-1] is original_handler

    def test_original_sigint_restored_even_on_exception(self, cfg):
        """SIGINT restauré même si une exception est levée dans le bloc."""
        proc = make_proc()
        original_handler = signal.SIG_DFL
        installed_handlers = []

        def fake_signal(signum, handler):
            if signum == signal.SIGINT:
                installed_handlers.append(handler)

        with patch("server.start_server", return_value=proc):
            with patch("server.stop_server"):
                with patch("server.signal.signal", side_effect=fake_signal):
                    with patch("server.signal.getsignal", return_value=original_handler):
                        with pytest.raises(ValueError):
                            with nexa_server(cfg):
                                raise ValueError("test")

        assert installed_handlers[-1] is original_handler

    def test_nexaservererror_propagates_from_start_server(self, cfg):
        """Si start_server lève NexaServerError, elle doit remonter."""
        with patch("server.start_server", side_effect=NexaServerError("modèle manquant")):
            with pytest.raises(NexaServerError, match="modèle manquant"):
                with nexa_server(cfg):
                    pass  # ne doit pas être atteint
