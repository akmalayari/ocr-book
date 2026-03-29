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
import socket

from config import Config

logger = logging.getLogger(__name__)


class NexaServerError(RuntimeError):
    pass


def _wait_for_server(proc, port: int, timeout_s: int) -> bool:

    url = f"http://127.0.0.1:{port}/v1/models"

    deadline = time.time() + timeout_s

    while time.time() < deadline:

        # Vérifier si le process est mort
        if proc.poll() is not None:

            stdout, stderr = proc.communicate()
            stderr_text = stderr.decode() if stderr else "(non disponible)"
            raise NexaServerError(
                "Le serveur s'est arrêté avant d'être prêt.\n"
                f"stderr:\n{stderr_text}"
            )

        try:
            r = requests.get(url, timeout=1)

            if r.status_code == 200:
                return True

        except requests.RequestException:
            pass

        time.sleep(0.5)

    return False


def start_server(cfg: Config) -> subprocess.Popen:
    """
    Lance `nexa serve` en arrière-plan et attend qu'il soit prêt.
    Retourne le processus.
    """
    cmd = ["nexa", "serve", cfg.model]

    logger.info("Démarrage serveur : %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if cfg.verbose else subprocess.DEVNULL,
        stderr=subprocess.PIPE if cfg.verbose else subprocess.DEVNULL,
    )

    if not _wait_for_server(proc,cfg.port, cfg.server_timeout_s):
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


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


@contextmanager
def nexa_server(cfg: Config):
    """
    Context manager : démarre le serveur à l'entrée, l'arrête à la sortie.
    Si un serveur tourne déjà sur le port configuré, s'y connecte directement
    sans en démarrer un nouveau (et sans l'arrêter à la sortie).

    Usage :
        with nexa_server(cfg) as proc:
            ...
    """
    if _is_port_in_use(cfg.port):
        logger.info("Serveur déjà actif sur le port %d, connexion directe.", cfg.port)
        yield None
        return

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
