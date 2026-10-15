"""
Reward shaping for the combat actor.

The reward of a transition is computed from the :class:`~l2_actor.world.StepEvents` collected
between two decisions. Every term is reported separately (``reward_terms`` in the stored
transition), so the learner can re-weight or recompute rewards later without new data.

The default weights are a starting point, not a tuned result:

==================  ===================================  ======================================
term                value                                intent
==================  ===================================  ======================================
``kill``            +1.0 per kill of our target          the goal of a fight
``death``           -1.0 per death                       dying is the worst outcome
``damage_dealt``    +0.3 x share of the target's HP      dense progress signal inside a fight;
                    dealt                                relative to the monster's HP, so its
                                                         scale does not grow with the level
``damage_taken``    -0.5 x share of own max HP lost      keep the character healthy
``loot``            +0.05 per item picked up             collect the drops
``step``            -0.001 per decision                  prefer short fights
==================  ===================================  ======================================
"""

from __future__ import annotations

from dataclasses import dataclass

from .world import StepEvents


@dataclass
class RewardConfig:
    """Reward weights; see the module docstring for the meaning of each term."""

    kill: float = 1.0
    death: float = -1.0
    damage_dealt: float = 0.3
    damage_taken: float = -0.5
    loot: float = 0.05
    step: float = -0.001

    def compute(self, events: StepEvents, max_hp: int) -> tuple[float, dict[str, float]]:
        """Return ``(reward, terms)`` for the events of one transition."""
        terms = {
            "kill": self.kill * events.kills,
            "death": self.death * events.deaths,
            "damage_dealt": self.damage_dealt * events.damage_dealt_share,
            "damage_taken": self.damage_taken * (events.damage_taken / max(max_hp, 1)),
            "loot": self.loot * events.items_picked,
            "step": self.step,
        }
        return sum(terms.values()), terms
