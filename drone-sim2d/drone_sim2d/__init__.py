from gym.envs.registration import register

register(id='drones2d-v0',entry_point='drone_sim2d.envs:DronesEnv')