import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./logs_sac/progress.csv")

# Examples (columns depend on what SB3 recorded for your run)
for col in ["rollout/ep_rew_mean", "train/critic_loss", "train/actor_loss", "train/ent_coef"]:
    if col in df.columns:
        plt.figure()
        plt.plot(df[col])
        plt.title(col)
        plt.xlabel("log index")
        plt.ylabel(col)

plt.show()
