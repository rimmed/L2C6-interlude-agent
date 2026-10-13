"""
Training-time augmentations for :class:`~l2_NN.dataloader.dataloder.Lineage2EventDataset` windows.

Each augmentation fights a concrete gap between the logged data and live play:

==========================  ====================================================================
augmentation                why
==========================  ====================================================================
``skill_shuffle``           the skill head is a pointer: slot order must carry no meaning
``mob_shuffle``             same for the target pointer (slots are distance-sorted in the logs)
``distractors``             add combat skills the character did not use, so the model learns to
                            *choose* by skill features instead of recognising a known panel
``time_scale``              the logs were recorded at TimeScale 1, the actors also run faster
``event_dropout``           live play sees a different rhythm of environment events (damage
                            messages) than the logs
``crop``                    do not rely on always having a full history window
``noise``                   small jitter on HP/MP/CP ratios and monster distances
==========================  ====================================================================

Only the training split is augmented; validation always sees the plain windows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AugmentConfig:
    """Probabilities and ranges of the window augmentations (0 disables one)."""

    skill_shuffle_p: float = 1.0
    mob_shuffle_p: float = 1.0
    distractor_p: float = 0.5
    max_distractors: int = 6
    time_scale_p: float = 0.5
    #: real ``delta_t`` is multiplied by a log-uniform factor from this range (< 1 = faster server)
    time_scale_range: tuple[float, float] = (0.2, 1.0)
    event_dropout_p: float = 0.3
    #: share of environment events (damage / effect messages) dropped from a window
    event_dropout_rate: float = 0.3
    crop_p: float = 0.2
    noise_p: float = 0.5
    noise_std: float = 0.02

    @classmethod
    def disabled(cls) -> AugmentConfig:
        return cls(0.0, 0.0, 0.0, 0, 0.0, (1.0, 1.0), 0.0, 0.0, 0.0, 0.0, 0.0)
