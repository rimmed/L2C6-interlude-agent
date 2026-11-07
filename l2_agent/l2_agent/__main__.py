"""
Run the agent: ``python -m l2_agent --account aiagent01 --password ... --name Aria``.

A new account gets a new Human Fighter (``--class-id``) named ``--name``; an existing one plays its character.
"""

from __future__ import annotations

import argparse
import logging
import sys

from l2_client import packets

from .brain import Brain, BrainConfig
from .explore import explore
from .session import Session, SessionConfig

#: CharacterCreate race of each starting class
RACES = {0: 0, 10: 0, 18: 1, 25: 1, 31: 2, 38: 2, 44: 3, 49: 3, 53: 4}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server-ip", default="192.168.1.200")
    p.add_argument("--account", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--name", help="Name of the character created when the account has none")
    p.add_argument("--class-id", type=int, default=0, help="Starting class of a new character (0 = Human Fighter)")
    p.add_argument("--female", action="store_true")
    p.add_argument("--model", default="checkpoints/agent/combat_policy.onnx")
    p.add_argument("--server-speed", type=float, default=1.0, help="The server's TimeScale")
    p.add_argument("--minutes", type=float, default=0.0, help="Stop after this long (0 = no limit)")
    p.add_argument("--explore", action="store_true",
                   help="Visit every NPC of Talking Island and write what was found (checkpoints/agent/explore_*)")
    p.add_argument("--gm-speed", type=int, default=0,
                   help="Ask for //gmspeed N after login (tests: the character must have GM access)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname).1s %(name)s: %(message)s", datefmt="%H:%M:%S")
    create = None
    if args.name:
        create = dict(name=args.name, race=RACES[args.class_id], female=args.female, class_id=args.class_id,
                      hair_style=1, hair_color=1, face=0)
    session = Session(SessionConfig(server_ip=args.server_ip, account=args.account, password=args.password,
                                    create=create, model=args.model, server_speed=args.server_speed))
    session.connect()
    if args.gm_speed:
        session.send(packets.build_bypass_build_cmd(f"gmspeed {args.gm_speed}"))
        session.idle(1.0)
    try:
        brain = Brain(session, BrainConfig())
        if args.explore:
            brain.tutorial.settle(2.0)
            explore(session, brain.talker, brain.trader, brain.cfg.area, brain.cfg.state_dir)
        else:
            brain.run(args.minutes)
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
