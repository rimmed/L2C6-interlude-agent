"""
Template features of one ``mobs_matrix`` row - shared by the training dataloader and the live client.

They describe *what the monster is* (from its datapack template, see
:func:`npc_data.get_monster_template`) relative to the character fighting it, so they carry over to
monsters and classes the model was never trained on - a raw npc id would not:

====================  ===========================================================================
``is_aggressive``     it attacks a character that comes close (``isAggressive``, default true)
``aggro_range_norm``  its aggro range / ``max_distance``, 0 for a passive monster
``is_ranged``         it hits from afar (a bow, a long attack range or long-range magic)
``is_undead``         race ``UNDEAD`` (Turn Undead, sacred weak points, holy weapons)
``toughness``         log10 of the hits (casts, for a mystic) the character needs to kill it, / 2, in [0, 1]
``weapon_affinity``   its weak point (> 0) or resistance (< 0) to the character's weapon type, in [-1, 1]
``def_bias``          ln(M.Def / P.Def), in [-1, 1]: > 0 magic does relatively worse than weapons
====================  ===========================================================================

Both sides call :func:`encode_mob_template`, so the features are identical by construction.
An unknown template encodes as all zeros.
"""

from __future__ import annotations

import math
from typing import Final

from .tools.data import npc_data

MOB_TEMPLATE_FEATURES: Final[tuple[str, ...]] = (
    "is_aggressive",
    "aggro_range_norm",
    "is_ranged",
    "is_undead",
    "toughness",
    "weapon_affinity",
    "def_bias",
)
MOB_TEMPLATE_DIM: Final[int] = len(MOB_TEMPLATE_FEATURES)

#: Interlude damage rough cuts: a hit does ~70 * P.Atk / P.Def, a nuke ~91 * sqrt(M.Atk) * power / M.Def
PHYSICAL_DAMAGE_FACTOR: Final[float] = 70.0
MAGIC_DAMAGE_FACTOR: Final[float] = 91.0
#: power of the nuke a mystic of level L has: ~11 + 1.3 * L (Wind Strike 12 at 1, Blaze 44 at 35, Prominence 108 at 74)
MAGIC_POWER_BASE: Final[float] = 11.0
MAGIC_POWER_PER_LEVEL: Final[float] = 1.3
#: ``toughness`` 1.0 = 10 ** this many hits
TOUGHNESS_LOG_SCALE: Final[float] = 2.0


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def encode_mob_template(npc_id: int | None, *, weapon_type: str, p_atk: float, m_atk: float, level: float,
                        max_distance: float) -> tuple[float, ...]:
    """
    Template features of the monster ``npc_id`` against a character.

    :param weapon_type: the character's weapon type (``item_data.get_weapon_type``, ``FIST`` unarmed);
        its name is the server's trait name (``BLUNT``, ``BOW`` ...)
    :param p_atk: / :param m_atk: the character's P. Atk. / M. Atk. (0 when unknown: ``toughness`` is 0)
    :param level: the character's level (the power of its nukes)
    :param max_distance: the distance scale of ``dist_norm``
    """
    template = npc_data.get_monster_template(int(npc_id)) if npc_id is not None else None
    if template is None:
        return (0.0,) * MOB_TEMPLATE_DIM

    # the character's way to fight it: nukes for a mystic (its M. Atk. outgrows its P. Atk.), weapons otherwise
    if m_atk > p_atk and template.m_def > 0:
        power = MAGIC_POWER_BASE + MAGIC_POWER_PER_LEVEL * level
        hits = template.max_hp * template.m_def / (MAGIC_DAMAGE_FACTOR * math.sqrt(m_atk) * power)
    elif p_atk > 0 and template.p_def > 0:
        hits = template.max_hp * template.p_def / (PHYSICAL_DAMAGE_FACTOR * p_atk)
    else:
        hits = 0.0
    toughness = _clip(math.log10(max(hits, 1.0)) / TOUGHNESS_LOG_SCALE, 0.0, 1.0)

    # damage of that trait is multiplied by 1 - value / 100 (Formulas.calcWeaponTraitBonus)
    affinity = _clip(-template.defence_traits.get(weapon_type or "FIST", 0.0) / 100.0, -1.0, 1.0)
    def_bias = (_clip(math.log(template.m_def / template.p_def), -1.0, 1.0)
                if template.p_def > 0 and template.m_def > 0 else 0.0)
    return (
        1.0 if template.aggressive else 0.0,
        min(template.aggro_range / max_distance, 1.0) if template.aggressive else 0.0,
        1.0 if template.ranged else 0.0,
        1.0 if template.race == "UNDEAD" else 0.0,
        toughness,
        affinity,
        def_bias,
    )
