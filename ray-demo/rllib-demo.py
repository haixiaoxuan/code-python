import ray
from ray import tune


ray.init()
tune.run(
    "PPO",
    stop={"episode_reward_mean": 200},
    config={
        "env": "CartPole-v0",
        # "num-gpus": 0,
        "num_workers": 10,
        "lr": tune.grid_search([0.001]),
        # "eager": False
    }
)