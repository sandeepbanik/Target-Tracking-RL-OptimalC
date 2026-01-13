import math


def quat_from_yaw(yaw: float):
    # Rotation about +Z
    return {"x": 0.0, "y": 0.0, "z": math.sin(yaw / 2.0), "w": math.cos(yaw / 2.0)}

def rot2d(dx: float, dy: float, yaw: float):
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * dx - s * dy, s * dx + c * dy)

def make_vehicle_scene_update(
    x: float, y: float, yaw: float,
    wheel_delta,  # array-like [fl, fr, rl, rr]
    target,       # array-like [tx, ty, ttheta] or at least [tx, ty]
    L: float, W: float,
    prefix=""
):
    # Cube sizes (tune as desired)
    body_size  = {"x": L,        "y": W,        "z": 0.2}
    wheel_size = {"x": L / 4.0,  "y": W / 6.0,  "z": 0.2}

    # Wheel centers in the vehicle body frame (adjust if your drawing convention differs)
    p_fl = (+L/2.0, +W/2.0)
    p_fr = (+L/2.0, -W/2.0)
    p_rl = (-L/2.0, +W/2.0)
    p_rr = (-L/2.0, -W/2.0)

    # Rotate offsets into world frame
    o_fl = rot2d(p_fl[0], p_fl[1], yaw)
    o_fr = rot2d(p_fr[0], p_fr[1], yaw)
    o_rl = rot2d(p_rl[0], p_rl[1], yaw)
    o_rr = rot2d(p_rr[0], p_rr[1], yaw)

    dfl, dfr, drl, drr = [float(v) for v in wheel_delta]

    # Target
    tx = float(target[0])
    ty = float(target[1])
    ttyaw = float(target[2]) if len(target) >= 3 else 0.0

    def cube_entity(eid, cx, cy, cyaw, size, color):
        return {
            "timestamp": None,          # optional
            "frame_id": "map",          # one global frame is enough
            "id": eid + prefix,
            "frame_locked": False,
            "cubes": [{
                "pose": {
                    "position": {"x": cx, "y": cy, "z": 0.0},
                    "orientation": quat_from_yaw(cyaw),
                },
                "size": size,
                "color": color,
            }]
        }

    return {
        "entities": [
            cube_entity("body", x, y, yaw, body_size,  {"r": 0.2, "g": 0.6, "b": 1.0, "a": 1.0}),
            cube_entity("wheel_fl", x + o_fl[0], y + o_fl[1], yaw + dfl, wheel_size, {"r": 0.1, "g": 0.1, "b": 0.1, "a": 1.0}),
            cube_entity("wheel_fr", x + o_fr[0], y + o_fr[1], yaw + dfr, wheel_size, {"r": 0.1, "g": 0.1, "b": 0.1, "a": 1.0}),
            cube_entity("wheel_rl", x + o_rl[0], y + o_rl[1], yaw + drl, wheel_size, {"r": 0.1, "g": 0.1, "b": 0.1, "a": 1.0}),
            cube_entity("wheel_rr", x + o_rr[0], y + o_rr[1], yaw + drr, wheel_size, {"r": 0.1, "g": 0.1, "b": 0.1, "a": 1.0}),
            cube_entity("target", tx, ty, ttyaw, {"x": 0.3, "y": 0.3, "z": 0.3}, {"r": 1.0, "g": 0.2, "b": 0.2, "a": 1.0}),
        ]
    }


EPISODE_SCHEMA = {
  "type": "object",
  "properties": {
    "episode_id": {"type": "integer"},
    "target": {
      "type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3
    },
    "t0": {"type": "number"}
  },
  "required": ["episode_id", "target", "t0"]
}

STATE_SCHEMA = {
  "type": "object",
  "properties": {
    "episode_id": {"type": "integer"},
    "t": {"type": "number"},
    "x": {"type": "number"},
    "y": {"type": "number"},
    "yaw": {"type": "number"},
    "v": {"type": "number"},
    "state": {"type": "array", "items": {"type": "number"}}
  },
  "required": ["episode_id", "t", "x", "y", "yaw"]
}

ACT_SCHEMA = {
  "type": "object",
  "properties": {
    "episode_id": {"type": "integer"},
    "t": {"type": "number"},
    "u_sac": {"type": "array", "items": {"type": "number"}},

    "delta_fl": {"type": "number"},
    "delta_fr": {"type": "number"},
    "delta_rl": {"type": "number"},
    "delta_rr": {"type": "number"},

    "v_fl": {"type": "number"},
    "v_fr": {"type": "number"},
    "v_rl": {"type": "number"},
    "v_rr": {"type": "number"}
  },
  "required": ["episode_id", "t", "u_sac"]
}

SCENEUPDATE_SCHEMA_MIN = {
  "type": "object",
  "properties": {
    "deletions": {"type": "array", "items": {"type": "object"}},
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "timestamp": {"type": "object"},
          "frame_id": {"type": "string"},
          "id": {"type": "string"},
          "frame_locked": {"type": "boolean"},
          "cubes": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "pose": {
                  "type": "object",
                  "properties": {
                    "position": {
                      "type": "object",
                      "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
                      "required": ["x", "y", "z"]
                    },
                    "orientation": {
                      "type": "object",
                      "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}, "w": {"type": "number"}},
                      "required": ["x", "y", "z", "w"]
                    }
                  },
                  "required": ["position", "orientation"]
                },
                "size": {
                  "type": "object",
                  "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
                  "required": ["x", "y", "z"]
                },
                "color": {
                  "type": "object",
                  "properties": {"r": {"type": "number"}, "g": {"type": "number"}, "b": {"type": "number"}, "a": {"type": "number"}},
                  "required": ["r", "g", "b", "a"]
                }
              },
              "required": ["pose", "size"]
            }
          }
        },
        "required": ["frame_id", "id"]
      }
    }
  },
  "required": ["entities"]
}

REWARD_SCHEMA = {
  "type": "object",
  "properties": {
    "episode_id": {"type":"integer"},
    "step": {"type":"integer"},
    "t": {"type":"number"},
    "reward": {"type":"number"},
    "done": {"type":"boolean"}
  },
  "required": ["episode_id","step","t","reward","done"]
}
TRAIN_SCHEMA = {
  "type": "object",
  "properties": {
    "episode_id": {"type":"integer"},
    "update_idx": {"type":"integer"},
    "t": {"type":"number"},
    "critic_loss": {"type":"number"},
    "actor_loss": {"type":"number"}
  },
  "required": ["episode_id","update_idx","t","critic_loss","actor_loss"]
}

