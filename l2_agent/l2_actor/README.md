# l2_actor — autonomous combat actor

Plays one agent character on the L2J server with the trained combat model and records its
combat trajectories (state, action, reward, next state, done) for later training. The actor only
executes the model's decisions; it has no policy of its own. It reuses the `l2_client` package:
packets, crypto, character/radar, and `CombatPolicy` for the observation encoding and the ONNX model.

## Running

From `l2_agent/`:

```bash
# Trained model on the CPU; a new random account, which means a new random agent character
uv run python -m l2_actor --model checkpoints/combat_policy.onnx --max-episodes 100

# Keep playing the same character
uv run python -m l2_actor --account rlactor7k2p9q --password 'actor!2026' --model checkpoints/combat_policy.onnx
```

Useful options (`--help` lists all of them):

| option | meaning |
|---|---|
| `--model` | **required**: the ONNX model from `l2_NN/export_onnx.py`; its `.json` (feature layout, skill panels) must sit next to it |
| `--device` | `cpu` (default) or `gpu` |
| `--server-speed` | the server's `TimeScale` from `General.ini` (default 1). **It must match the server**, otherwise cooldowns and cast times are wrong |
| `--max-episodes` / `--max-steps` / `--max-minutes` | stop conditions (0 = no limit); Ctrl+C also stops cleanly |
| `--output-dir` | where the files go (default `trajectories/`) |
| `--flush-every` | flush the buffer every N transitions (it also flushes at every episode end) |
| `--no-next-state` | do not store `next_state`; it is the `state` of the next record of the episode, and storing it doubles the size |
| `-v` | log every decision |

Several actors can run at once: each uses its own account and writes its own files.

## What happens

1. **Login.** An unknown account is created by the login server (`AutoCreateAccounts`). The
   RL-environment game server (`EnableRLAgentAutoSetup`) then gives it a random character and,
   on login, sets it up: random level, class transfers, skills, gear of the right grade, shots and
   potions, and a teleport to the farm zone of its level. The actor waits until this setup has landed.
2. **Combat loop.** Every `--decision-interval` seconds the actor builds the observation (the same
   tensors the model was trained on), runs the model and executes its decision. No decision
   is taken while the character's own cast is running or while it walks to loot, because a new
   command would cancel it.
3. **Episodes.** An episode ends when the current target dies (`kill`) or the character dies
   (`death`). On death the actor does nothing: the server revives the character in its farm zone
   with full HP/MP. Only if that does not happen within `--respawn-timeout` does the actor request
   the standard town respawn.
4. **Storage.** Every decision becomes a `Transition`. Its reward is computed from what happened
   until the next decision. Transitions go through `TransitionBuffer.add()`.

## Data layout

`transitions_<actor>.jsonl` holds one transition per line:

```jsonc
{
  "actor_id": "rlactor88cd40", "episode_id": 3, "step": 5, "t": 1791892367.123,
  "state": {                                  // model input, left-padded window of L=16 events
    "player_state": [[...8]],                 // L x 8   HP/MP/CP ratios, combat flag, level, speeds
    "delta_t": [...],                         // L       log(1 + seconds since the previous event)
    "mobs_matrix": [[[...4]]],                // L x 10 x 4  nearest monsters: distance, level diff, target flag, padding flag
    "skills_matrix": [[[...4]]],              // L x 16 x 4  skill panel: available, ready, cooldown, enough MP
    "seq_mask": [...]                         // L       true for real events
  },
  "action": {"type": "used_skill", "type_id": 3, "pointer_idx": 0, "skill_slot": 1,
             "mob_object_id": 268451514, "skill_id": 1184, "confidence": 0.81,
             "model": "combat_policy.onnx", "executed": "UseSkill Ice Bolt"},
  "action_mask": {"types": [0, 1, 2, 3], "n_mobs": 4, "skill_slots": [0, 1]},
  "reward": 0.112, "reward_terms": {"kill": 0, "death": 0, "damage_dealt": 0.113, "damage_taken": 0, "loot": 0, "step": -0.001},
  "next_state": { ...same as state... },      // null for the last step of an episode
  "done": false, "truncated": false, "done_reason": null,     // "kill" / "death" on the last step
  "info": {"character": "RLxzt4tp", "level": 9, "class_id": 18, "hp_ratio": 0.93, "mp_ratio": 1.0,
           "pos": [45980, 41099, -3499], "target_id": 268451514, "loot_on_ground": 0}
}
```

* `type_id`, `pointer_idx` and `skill_slot` are indices into the three model heads, so a learner can
  use them directly as action labels. `action_mask` says which outputs were allowed at that step.
* `reward_terms` keeps every component (see `reward.py`), so rewards can be re-weighted later
  without collecting new data.
* `truncated: true` marks a step cut off by stopping the actor. It is not a terminal state.

`episodes_<actor>.jsonl` holds one summary per episode: `episode_id`, `reason`, `steps`, `return`,
`duration_s`, character, level and class.

## Plugging in Redis

The actor only uses the abstract `TransitionBuffer` (`buffer.py`): `add(transition)`,
`end_episode()`, `flush()`, `close()`. A Redis buffer subclasses it, for example by doing
`XADD`/`RPUSH` of `transition.to_record()` in `flush()`, and is passed in as `Actor(cfg, buffer=...)`.

## Modules

| module | role |
|---|---|
| `actor.py` | CLI, setup, combat loop, episodes, transitions |
| `world.py` | game state from server packets; reward events; kill / death / revive / teleport signals |
| `executor.py` | decisions → packets, with the safeguards found in live tests |
| `reward.py` | reward weights and computation |
| `buffer.py` | `Transition`, `TransitionBuffer`, `JsonlBuffer` |
| `../l2_client/connection.py` | login + game-server handshake, non-blocking packet pump |

## Known limitations

* Skills are encoded as slots of the panel of a *training* character. A class whose combat skills
  are not on any training panel (e.g. an Orc Shaman) cannot cast them; it fights with its weapon.
* There is no pathfinding. Monsters far above or below the character are ignored, and a target
  that takes no hit for 15 s is dropped for 60 s.
* Monster positions on the radar are their spawn positions (NPC movement packets are not parsed).
