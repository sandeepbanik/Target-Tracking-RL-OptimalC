import json
import numpy as np
import matplotlib.pyplot as plt

from mcap.reader import make_reader

MCAP_PATH = "sac_training_full.mcap"
TOPIC = "/rl/transition"

# Choose which element(s) of obs correspond to x,y,theta in your environment.
# Example: obs = [x, y, theta, ...]
IDX_X, IDX_Y, IDX_THETA = 0, 1, 2

xs, ys, thetas, steps = [], [], [], []

with open(MCAP_PATH, "rb") as f:
    reader = make_reader(f)
    for schema, channel, message in reader.iter_messages(topics=[TOPIC]):
        # message.data is bytes (JSON)
        d = json.loads(message.data)

        obs = np.array(d["obs"], dtype=float)
        xs.append(obs[IDX_X])
        ys.append(obs[IDX_Y])
        thetas.append(obs[IDX_THETA])
        steps.append(d.get("step", None))

xs = np.array(xs)
ys = np.array(ys)

plt.figure()
plt.plot(xs, ys)
plt.axis("equal")
plt.title("Replayed trajectory from MCAP (/rl/transition)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()