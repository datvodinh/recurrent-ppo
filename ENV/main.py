import env

env.make("Catan")
print(env.run(env.agent_random, num_game= 10000, level=0))