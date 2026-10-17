"""
Lineage2 C6 Interlude -- summon (servitor / pet) skills and commands from the server XML datapack.

A summon's skills are not on the owner's skill list: the owner commands them through
RequestActionUse (0x45) with an *action id*. The server maps an action id to a skill of the
summon's NPC template (``RequestActionUse.java``): most actions name a skill holder of the
template's ``<parameters>`` (``<skill name="DDMagic" id="4137" level="6" />``), a few a skill id
directly. The template's ``<param name="step..._action..">`` values list the actions its client
panel shows.

The MIT License (MIT)

Copyright (c) 2026 Viacheslav Reutskyi <viacheslav.reutskyi@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .npc_data import DEFAULT_NPC_XML_PATH

#: RequestActionUse ids of the summon commands (servitor, pet).
ATTACK_ACTIONS = (22, 16)
FOLLOW_ACTIONS = (21, 15)
STOP_ACTIONS = (23, 17)

#: RequestActionUse skill actions (``RequestActionUse.java``): action id -> (skill holder name of the
#: summon's template or a skill id, cast on the owner instead of the target).
ACTION_SKILLS: dict[int, tuple[str | int, bool]] = {
    32: ("DDMagic", False), 36: ("RangeDebuff", False), 39: ("RangeDD", False), 41: (4230, False),
    42: ("HealMagic", False), 43: ("DDMagic", False), 44: ("DDMagic", False), 45: ("HealMagic", True),
    46: ("DDMagic", False), 47: ("DDMagic", False), 48: ("DDMagic", False), 1000: (4079, False),
    1003: ("PhysicalSpecial", False), 1004: ("Buff", True), 1005: ("DDMagic", False), 1006: ("Heal", True),
    1007: ("Buff1", True), 1008: ("Buff2", True), 1009: ("DDMagic", False), 1010: ("Buff1", True),
    1011: ("Buff2", True), 1012: ("DDMagic", False), 1013: ("DeBuff1", False), 1014: ("DeBuff2", False),
    1015: ("Heal", False), 1016: ("PhysicalSpecial1", False), 1017: ("PhysicalSpecial2", False),
    1031: ("PhysicalSpecial1", False), 1032: ("PhysicalSpecial2", False), 1033: ("PhysicalSpecial3", False),
    1034: ("PhysicalSpecial1", False), 1035: ("PhysicalSpecial2", False), 1036: ("PhysicalSpecial1", False),
    1037: ("PhysicalSpecial2", False), 1038: ("PhysicalSpecial3", False), 1039: (5110, False), 1040: (5111, False),
}


@dataclass(frozen=True)
class PetSkill:
    """A skill the owner can order its summon to cast."""

    action_id: int
    skill_id: int
    level: int
    #: cast on the owner (heals, buffs) instead of the owner's target
    on_owner: bool


_templates: dict[int, list[PetSkill]] = {}
_loaded = False

_NPC = re.compile(r'<npc id="(\d+)"[^>]*type="Pet"[^>]*>(.*?)</npc>', re.S)
_PARAMS = re.compile(r"<parameters>(.*?)</parameters>", re.S)
_SKILL_HOLDER = re.compile(r'<skill name="(\w+)" id="(\d+)" level="(\d+)"')
_ACTION_PARAM = re.compile(r'<param name="step\w*action\w*" value="(\d+)"')
_SKILL_LIST = re.compile(r'<skill id="(\d+)" level="(\d+)"')


def load_summon_data(path: str = DEFAULT_NPC_XML_PATH) -> None:
    """Read the summon templates (``type="Pet"``: servitors and pets) once."""
    global _loaded
    if _loaded:
        return
    for name in sorted(os.listdir(path)):
        if not name.endswith(".xml"):
            continue
        with open(os.path.join(path, name), encoding="utf-8") as f:
            text = f.read()
        for npc in _NPC.finditer(text):
            npc_id, body = int(npc.group(1)), npc.group(2)
            params = _PARAMS.search(body)
            holders = {m.group(1): (int(m.group(2)), int(m.group(3))) for m in _SKILL_HOLDER.finditer(params.group(1))} \
                if params else {}
            levels = {int(m.group(1)): int(m.group(2)) for m in _SKILL_LIST.finditer(body)}
            skills = []
            for action_id in (int(m.group(1)) for m in _ACTION_PARAM.finditer(params.group(1))) if params else ():
                spec = ACTION_SKILLS.get(action_id)
                if spec is None:
                    continue
                ref, on_owner = spec
                if isinstance(ref, int):
                    skill = (ref, levels.get(ref, 1))
                else:
                    skill = holders.get(ref)
                if skill is not None:
                    skills.append(PetSkill(action_id, skill[0], skill[1], on_owner))
            _templates[npc_id] = skills
    _loaded = True


def pet_skills(npc_id: int) -> list[PetSkill]:
    """The skills of a summon template the owner can order (``[]`` for an unknown template or one without)."""
    if not _loaded:
        load_summon_data()
    return list(_templates.get(int(npc_id), ()))


def is_summon_template(npc_id: int) -> bool:
    if not _loaded:
        load_summon_data()
    return int(npc_id) in _templates
