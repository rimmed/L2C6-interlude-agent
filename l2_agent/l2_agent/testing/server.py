"""
The game server's host over SSH, for the tests: run commands, copy files, change an offline character in the
database (position, access level).

``L2_HOST`` is the SSH target (default ``l2server@192.168.1.200``) and ``L2_SSH`` / ``L2_SCP`` the commands (default
``ssh`` / ``scp``; e.g. a wrapper that answers the password through ``SSH_ASKPASS``). The database login is read on
the server from ``~/server/game/config/Database.ini``.
"""

from __future__ import annotations

import os
import shlex
import subprocess

HOST = os.environ.get("L2_HOST", "l2server@192.168.1.200")
SSH = shlex.split(os.environ.get("L2_SSH", "ssh"))
SCP = shlex.split(os.environ.get("L2_SCP", "scp"))
DB_HOST = os.environ.get("L2_DB_HOST", "192.168.1.201")

_SQL = """cd ~/server
U=$(awk -F" = " "/^Login /{print \\$2}" game/config/Database.ini)
export MYSQL_PWD=$(awk -F" = " "/^Password /{print \\$2}" game/config/Database.ini)
mysql -h %s -u $U l2jdb -N -e "%s"
"""


def run(command: str, timeout: float = 120.0) -> str:
    """Run a shell command on the server; its standard output."""
    done = subprocess.run([*SSH, HOST, command], capture_output=True, text=True, timeout=timeout)
    if done.returncode:
        raise RuntimeError(f"ssh {command!r} failed: {done.stderr.strip()}")
    return done.stdout


def copy_to(local: str, remote: str, timeout: float = 300.0) -> None:
    subprocess.run([*SCP, "-q", local, f"{HOST}:{remote}"], check=True, timeout=timeout)


def sql(statement: str) -> str:
    """Run one SQL statement on the game database; its output."""
    done = subprocess.run([*SSH, HOST, "sh -s"], input=_SQL % (DB_HOST, statement.replace('"', '\\"')),
                          capture_output=True, text=True, timeout=120)
    if done.returncode:
        raise RuntimeError(f"sql failed: {done.stderr.strip()}")
    return done.stdout


def set_position(name: str, x: int, y: int, z: int, access_level: int | None = None) -> bool:
    """Move an offline character (and set its access level); ``False`` when it is online or unknown."""
    access = f", accesslevel={int(access_level)}" if access_level is not None else ""
    out = sql(f"UPDATE characters SET x={int(x)}, y={int(y)}, z={int(z)}{access} "
              f"WHERE char_name='{name}' AND online=0; SELECT ROW_COUNT();")
    return out.strip().endswith("1")


def set_access(name: str, access_level: int) -> bool:
    """Set an offline character's access level (0 player, 30 General GM, 100 master); ``False`` when online."""
    out = sql(f"UPDATE characters SET accesslevel={int(access_level)} WHERE char_name='{name}' AND online=0; "
              "SELECT ROW_COUNT();")
    return out.strip().endswith("1")
