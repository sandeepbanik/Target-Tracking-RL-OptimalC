import json
import time
from typing import Any, Dict, Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from mcap.writer import Writer


class MCAPMetricsCallback(BaseCallback):
    """
    Writes training metrics + episode stats to an MCAP file as JSON messages.
    Creates three channels:
      - /rl/metrics : periodic training metrics (losses, ent coef, etc. if available)
      - /rl/episode : episode return/length from Monitor wrapper
      - /rl/marker  : optional event markers (start/end)

    """
    def __init__(self,
                 mcap_path: str, 
                 metrics_every_n_steps: int = 1000,
                 transitions_every_n_steps: int = 1,  
                 verbose: int = 0):
        super().__init__(verbose=verbose)
        self.mcap_path = mcap_path  # Storing path for the MCAP file
        self.metrics_every_n_steps = metrics_every_n_steps  # Frequency of metrics logging
        self.transitions_every_n_steps = transitions_every_n_steps  # Frequency of transition logging


        self._f = None
        self._writer: Optional[Writer] = None

        self._schema_id = None
        self._ch_metrics = None
        self._ch_episode = None
        self._ch_marker = None
        self._ch_transition = None

        self._seq = 0

    def _now_ns(self) -> int:
        return int(time.time() * 1e9)

    def _register_json_channel(self, topic: str) -> int:
        # A minimal JSON schema.
        # Schema is intentionally permissive to avoid tight coupling.
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "GenericJSON",
            "type": "object",
            "additionalProperties": True,
        }
        schema_id = self._writer.register_schema(
            name="GenericJSON",
            encoding="jsonschema",
            data=json.dumps(schema).encode("utf-8"),
        )
        ch_id = self._writer.register_channel(
            topic=topic,
            message_encoding="json",
            schema_id=schema_id,
        )
        return ch_id

    def _write(self, channel_id: int, msg: Dict[str, Any]) -> None:
        t = self._now_ns()
        self._writer.add_message(
            channel_id=channel_id,
            log_time=t,
            publish_time=t,
            data=json.dumps(msg).encode("utf-8"),
            sequence=self._seq,
        )
        self._seq += 1

    def _jsonify(self, x: Any) -> Any:
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, (list, tuple)):
            return [self._jsonify(v) for v in x]
        return x
    
    def _on_training_start(self) -> None:
        """
        Start of logging to MCAP file during training
        """
        self._f = open(self.mcap_path, "wb")
        self._writer = Writer(self._f) # Opens a Binary writer.

        # Channels
        self._ch_metrics = self._register_json_channel("/rl/metrics")
        self._ch_episode = self._register_json_channel("/rl/episode")
        self._ch_marker = self._register_json_channel("/rl/marker")
        self._ch_transition = self._register_json_channel("/rl/transition")

        self._write(self._ch_marker, {
            "event": "training_start",
            "algo": self.model.__class__.__name__,
        })

    def _on_step(self) -> bool:
        # Episode stats from Monitor: info.get("episode") contains {"r":..., "l":..., "t":...}
        infos = self.locals.get("infos", None)
        if infos is not None:
            for info in infos:
                ep = info.get("episode", None)
                if ep is not None:
                    self._write(self._ch_episode, {
                        "step": int(self.num_timesteps),
                        "episode_return": float(ep.get("r")),
                        "episode_length": int(ep.get("l")),
                        "episode_time_sec": float(ep.get("t", 0.0)),
                    })

        # Periodic transition snapshots.
        if (self.num_timesteps % self.transitions_every_n_steps) == 0:
            obs = self.locals.get("obs", None)
            actions = self.locals.get("actions", None)
            rewards = self.locals.get("rewards", None)
            dones = self.locals.get("dones", None)
            new_obs = self.locals.get("new_obs", None)

            # These keys exist for SB3 OffPolicyAlgorithm training loop in most setups.
            if obs is not None and actions is not None and new_obs is not None:
                obs_a = np.array(obs)
                act_a = np.array(actions)
                rew_a = np.array(rewards) if rewards is not None else None
                done_a = np.array(dones) if dones is not None else None
                new_obs_a = np.array(new_obs)

                # Handle both non-vectorized and vectorized envs
                if obs_a.ndim >= 2 and act_a.ndim >= 2:
                    n_env = obs_a.shape[0]
                    for i in range(n_env):
                        self._write(self._ch_transition, {
                            "step": int(self.num_timesteps),
                            "env_id": int(i),
                            "obs": self._jsonify(obs_a[i]),
                            "action": self._jsonify(act_a[i]),
                            "reward": self._jsonify(rew_a[i] if rew_a is not None else None),
                            "done": bool(done_a[i]) if done_a is not None else False,
                            "next_obs": self._jsonify(new_obs_a[i]),
                        })
                else:
                    self._write(self._ch_transition, {
                        "step": int(self.num_timesteps),
                        "env_id": 0,
                        "obs": self._jsonify(obs_a),
                        "action": self._jsonify(act_a),
                        "reward": self._jsonify(rew_a if rew_a is not None else None),
                        "done": bool(done_a) if done_a is not None else False,
                        "next_obs": self._jsonify(new_obs_a),
                    })


        # Periodic training metrics snapshot
        if (self.num_timesteps % self.metrics_every_n_steps) == 0:
            metrics = {"step": int(self.num_timesteps)}

            # SB3 internal logger values (contains last dumped values when available).
            # This is best-effort; some keys exist only after training updates begin.
            try:
                name_to_value = getattr(self.model.logger, "name_to_value", {})
                for k, v in name_to_value.items():
                    # Filter to typical training curves
                    if k.startswith(("train/", "rollout/", "time/")):
                        try:
                            metrics[k] = float(v)
                        except Exception:
                            pass
            except Exception:
                pass

            self._write(self._ch_metrics, metrics)

        return True

    def _on_training_end(self) -> None:
        """
        End of logging to MCAP file during training.
        """
        if self._writer is not None and self._ch_marker is not None:
            self._write(self._ch_marker, {"event": "training_end", "final_step": int(self.num_timesteps)})

        if self._writer is not None:
            self._writer.finish()
        if self._f is not None:
            self._f.close()
