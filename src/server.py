"""
server.py — Démarrage et arrêt du serveur Nexa (context manager)
"""

import subprocess
import time
import logging
import signal
import sys
from contextlib import contextmanager

import requests

from config import Config

logger = logging.getLogger(__name__)


class NexaServerError(RuntimeError):
    pass


def _wait_for_server(port: int, timeout_s: int) -> bool:
    """Attendre que le serveur réponde sur /v1/models."""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False


def start_server(cfg: Config) -> subprocess.Popen:
    """
    Lance `nexa serve` en arrière-plan et attend qu'il soit prêt.
    Retourne le processus.
    """
    cmd = [
        "nexa", "serve",
        "--model", cfg.model,
        "--port",  str(cfg.port),
    ]
    logger.info("Démarrage serveur : %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if cfg.verbose else subprocess.DEVNULL,
        stderr=subprocess.PIPE if cfg.verbose else subprocess.DEVNULL,
    )

    if not _wait_for_server(cfg.port, cfg.server_timeout_s):
        proc.kill()
        raise NexaServerError(
            f"Le serveur Nexa n'a pas répondu après {cfg.server_timeout_s}s. "
            f"Vérifiez que le modèle '{cfg.model}' est bien téléchargé "
            f"(nexa pull {cfg.model})."
        )

    logger.info("Serveur Nexa prêt sur le port %d.", cfg.port)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    """Arrêt propre du serveur."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        logger.info("Serveur Nexa arrêté.")


@contextmanager
def nexa_server(cfg: Config):
    """
    Context manager : démarre le serveur à l'entrée, l'arrête à la sortie.

    Usage :
        with nexa_server(cfg) as proc:
            ...
    """
    proc = start_server(cfg)

    # Arrêt propre sur Ctrl+C
    original_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(sig, frame):
        logger.info("Interruption reçue, arrêt du serveur…")
        stop_server(proc)
        signal.signal(signal.SIGINT, original_sigint)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        yield proc
    finally:
        stop_server(proc)
        signal.signal(signal.SIGINT, original_sigint)
