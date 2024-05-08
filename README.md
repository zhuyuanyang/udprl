# Discretizing Continuous Action Space with Unimodal Probability Distributions for On-Policy Reinforcement Learning

This code base contains all instructions and codes necessary to reproduce the results in the paper. We provide PPO with Gaussian, Gaussian + tanh, Discrete, Ordinal and Beta policy. 

On-Policy Optimization Baselines offer a suite of on-policy optimization algorithms, built on top of OpenAI [baselines](https://github.com/openai/baselines). This repository also contains wrappers necessary for discretizing continuous action space with unimodal probability distributions for on-policy reinforcement learning.


# Dependencies
Need to install OpenAI baselines and tensorflow.

```
pip install baselines
pip install --upgrade --ignore-installed https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.6.0-cp35-none-linux_x86_64.whl
```

## Examples

To run PPO with discrete policy (with K=11 actions bins per dimension)

```
python ppo_discrete/run_ppo_discrete.py --bins 11 --env Hopper-v1 --seed 100
```




## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
