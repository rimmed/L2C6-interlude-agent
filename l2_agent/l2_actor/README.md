# l2_actor — autonomous combat actor

Plays one agent character on the L2J server with the trained combat model and records its
combat trajectories (state, action, reward, next state, done) for training: to local `.jsonl`
files, or to Redis for the RL trainer `l2_NN/train_rl.py` (see [RL training](#rl-training-with-redis)). The actor only
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
| `--model` | **required**: the ONNX model from `l2_NN/export_onnx.py` (its `.json` with the feature layout must sit next to it), or `redis` for the trainer's latest policy |
| `--redis` | Redis URL; send the trajectories to the RL trainer instead of `.jsonl` |
| `--greedy` / `--sample` | take the most likely action / sample it (default: sample with `--redis`, greedy without) |
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
    "skills_matrix": [[[...17]]],             // L x 64 x 17 skill panel: datapack features + ready, cooldown, enough MP
    "seq_mask": [...]                         // L       true for real events
  },
  "action": {"type": "used_skill", "type_id": 3, "pointer_idx": 0, "skill_slot": 1,
             "mob_object_id": 268451514, "skill_id": 1184, "confidence": 0.81, "logp": -0.43,
             "model": "combat_policy.onnx", "model_version": null, "executed": "UseSkill Ice Bolt"},
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
* `logp` is the log-probability of the whole decision under the policy that made it (action type,
  plus the pointer for actions that use a monster, plus the skill slot for `used_skill`).
* `truncated: true` marks a step cut off by stopping the actor. It is not a terminal state.

`episodes_<actor>.jsonl` holds one summary per episode: `episode_id`, `reason`, `steps`, `return`,
`duration_s`, character, level and class.

## RL training with Redis

```text
 actors (any machines)                Redis                         trainer (GPU)
 python -m l2_actor  --RPUSH segments--> l2:replay   --LPOP-->   l2_NN/train_rl.py
   --model redis     <--MGET (poll)---- l2:model:*  <--MULTI--   (V-trace + BC on demos)
```

1. Start Redis reachable by every actor (`bind * -::*`, port 6379), then the trainer. It
   publishes the starting policy (version 1) before anything else:

   ```bash
   uv run l2_NN/train_rl.py --init checkpoints/best_model.pth --redis redis://localhost:6379/0
   ```

2. Start the actors on any machine. Each one waits for the published policy, plays it with
   sampled actions and loads every newer version without leaving the game:

   ```bash
   uv run python -m l2_actor --redis redis://<trainer-host>:6379/0 --model redis --server-speed 5
   ```

3. Stop the trainer with Ctrl+C or SIGTERM; it writes `checkpoints/rl/latest.pth`. Continue with
   `--resume checkpoints/rl/latest.pth`. `export_onnx.py -c checkpoints/rl/best.pth` exports an RL
   checkpoint like a behaviour-cloning one.

**Many actors on one machine.** `python -m l2_actor.fleet` runs `--actors` actors in worker
processes of `--per-proc` threads (default 10), one onnxruntime thread per model session, logins
spaced `--stagger` seconds apart, crashed actors restarted with the same account. Accounts are
`<prefix>0000`, `<prefix>0001`, ... so a restarted fleet plays the same characters:

```bash
uv run python -m l2_actor.fleet --actors 800 --redis redis://<trainer-host>:6379/0 --model redis --server-speed 5
```

It prints one summary line every `--report` seconds (actors online, episodes, kills / deaths,
transitions, model versions in use); per-process logs go to `<output-dir>/fleet_logs/`. On a
20-core machine 10 actors per process ran 800 actors at about half the CPU; one process per 25
actors already lost connections at 600 (the threads starve on the GIL).

**What travels.** An actor sends a *segment* (up to `--segment-len` steps, 64 by default) of one
episode as soon as it is full, or when the episode ends. A segment holds the model inputs of every
step, the chosen indices, the behaviour `logp`, the rewards and the masks (`replay.py` has the full
layout). It is packed with `numpy.savez_compressed` (float16, no pickle) and takes about 0.2 KB per
step on the wire. The actor trims `l2:replay` to `--max-queue` segments, so Redis stays bounded when
no trainer runs. The policy lives in `l2:model:onnx` / `l2:model:meta` / `l2:model:version`.

**What the trainer does** (`train_rl.py --help` lists every knob):

| step | default |
|---|---|
| replay buffer of the newest transitions, as whole segments | `--buffer-steps 50000` (about 2 GB RAM decoded) |
| drop segments played with a policy more than N versions old | `--max-staleness 20` |
| learn only while trained / received transitions stays below | `--max-replay-ratio 8` |
| first updates train only the new value head | `--value-warmup 200` |
| V-trace actor-critic on batches of segments, AdamW | `--batch-segments 32 --lr 3e-5 --gamma 0.99` |
| behaviour cloning on the human demonstrations, weight decays | `--demo-batch 256 --bc-start 1.0 --bc-end 0.05 --bc-decay 20000` |
| publish the policy | `--publish-every 100` updates |
| `latest.pth`, `best.pth` (best mean return of the last 100 episodes) | `--save-every 500` |

`<save-dir>/metrics.jsonl` gets one line per `--log-every` updates: losses, entropy, importance
weights (`rho_mean`, `rho_clipped`, `kl_approx`), buffer, replay ratio, episodes, kills/deaths and
the mean return.

**Health signs.** `kl_approx` near 0 and `rho_mean` near 1 mean the actors play the policy the
trainer thinks they play. A growing `rho_clipped` means the actors lag behind: publish more often
or lower the learning rate. A replay ratio pinned at the limit means the trainer waits for data:
add actors.

## Modules

| module | role |
|---|---|
| `actor.py` | CLI, setup, combat loop, episodes, transitions |
| `world.py` | game state from server packets; reward events; kill / death / revive / teleport signals |
| `executor.py` | decisions → packets, with the safeguards found in live tests |
| `reward.py` | reward weights and computation |
| `buffer.py` | `Transition`, `TransitionBuffer`, `JsonlBuffer` |
| `replay.py` | Redis keys, segment format, `RedisBuffer`, model publish / fetch |
| `../l2_client/connection.py` | login + game-server handshake, non-blocking packet pump |

## Known limitations

* There is no pathfinding. Monsters far above or below the character are ignored, and a target
  that takes no hit for 15 s is dropped for 60 s.
* Monster positions on the radar are their spawn positions (NPC movement packets are not parsed).
