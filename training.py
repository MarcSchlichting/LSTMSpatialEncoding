import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import drone_sim2d
import numpy as np
from datetime import datetime
import collections

device = "cpu"

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_var, action_std):      #state = 2+2+1+240 = 245
        super(ActorCritic, self).__init__()        
        ######### Actor Layers #########
        self.lstm1_a = nn.LSTM(input_size=3,hidden_size=63)
        self.fc2_a = nn.Linear(64,64)
        self.fc3_a = nn.Linear(64,2)
        self.h0_a = torch.zeros(1,1,63)
        self.c0_a = torch.zeros(1,1,63)                    
                

        ######### Critic Layers #########
        self.lstm1_c = nn.LSTM(input_size=3,hidden_size=63)
        self.fc2_c = nn.Linear(64,64)
        self.fc3_c = nn.Linear(64,1)
        self.h0_c = torch.zeros(1,1,63)
        self.c0_c = torch.zeros(1,1,63)
        

        #self.critic = nn.Sequential(critic_net)
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    ###add actor definition
    def actor(self,x,y):
        x = torch.FloatTensor(np.array(x)).view(-1,1,3)
        x, _ = self.lstm1_a(x,(self.h0_a,self.c0_a))
        x = x[-1,:,:]
        x = x.view(-1)
        x = torch.tanh(x)
           
        z = torch.cat((x,torch.FloatTensor(np.array([y]))),dim=-1)
        z = self.fc2_a(z)
        z = torch.tanh(z)
        z = self.fc3_a(z)
        z = torch.tanh(z)
        return z

    def critic(self,x,y):
        x = torch.FloatTensor(np.array(x)).view(-1,1,3)
        x, _ = self.lstm1_c(x,(self.h0_c,self.c0_c))
        x = x[-1,:,:]
        x = x.view(-1)
        x = torch.tanh(x)
           
        z = torch.cat((x,torch.FloatTensor(np.array([y]))),dim=-1)
        z = self.fc2_c(z)
        z = torch.tanh(z)
        z = self.fc3_c(z)
        z = torch.tanh(z)
        return z

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, torch.diag(self.action_var).to(device))
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):
        action_mean = []
        state_value = []
        no_vehicles = len(state)
        for idx_1 in range(no_vehicles):
            action_mean.append(self.actor(state[idx_1][1],state[idx_1][0]))
            state_value.append(self.critic(state[idx_1][1],state[idx_1][0]))
        action_mean = torch.stack(action_mean).view(-1,1,2)
        state_value = torch.stack(state_value).view(-1,1,1)
        
        #action_mean = self.actor(state)
        dist = MultivariateNormal(torch.squeeze(action_mean), torch.diag(self.action_var))
        
        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        #state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(device)
        #filename = "PPO_Continuous_drones2d-v0_288000_1562915157"
        #self.policy.load_state_dict(torch.load("./models/"+filename+".pth"))
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                              lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(device)
        #self.policy_old.load_state_dict(torch.load("./models/"+filename+".pth"))
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = memory.states
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
     
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1.float(), surr2.float()) + 0.5*self.MseLoss(state_values, rewards.float()) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

class AgentHandler:
    def __init__(self):
        self.agentmemory = AgentMemory()

    def state_handler(self,state):
        no_vehicles = len(state)
        new_states = []
        for idx_1 in range(no_vehicles):
            new_states = state
            if len(state[idx_1][1]) == 0:
                new_states[idx_1] = (state[idx_1][0],[np.array([0,0,0])])
        return new_states

    def del_done(self,response):

        no_vehicles = len(response)
        del_list = []
        for idx_1 in range(no_vehicles):
            if response[idx_1][2]==True:
                del_list.append(idx_1)
        del_list.sort(reverse=True)
        for idx_2 in range(len(del_list)):
            del response[del_list[idx_2]]
        return response

    def select_action(self,state,actor):
        self.action_var = torch.full((2,), 0.6*0.6).to(device)   #manually change action_dim action_std
        no_vehicles = len(state)
        action_list = []
        for idx_1 in range(no_vehicles):
            state1 = state[idx_1]
            target1 = state1[0]
            other_vehicles1 = state1[1] 
            action_mean = actor(other_vehicles1,target1)
            dist = MultivariateNormal(action_mean, torch.diag(self.action_var).to(device))      ##ATTENTION: manually change variance in torch.diag(var)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            self.agentmemory.memory_list[idx_1].states.append(state1)
            self.agentmemory.memory_list[idx_1].actions.append(action)
            self.agentmemory.memory_list[idx_1].logprobs.append(action_logprob)

            action_list.append(action.detach().cpu().data.numpy().flatten())
            
        return action_list   

    def response_eval(self,response):
        no_vehicles = len(response)
        state_list = []
        reward_list = []
        done_list = []
        for idx_1 in range(no_vehicles):            
            state = response[idx_1][0]   #only works one additional vehicle
            if len(state[1]) == 0:
                state = (state[0],[np.array([0,0,0])])
            reward = response[idx_1][1]
            done = response[idx_1][2] 

            state_list.append(state)
            reward_list.append(reward)
            done_list.append(done)           

        return state_list,reward_list,done_list

    def reward_memory_append(self,reward):
        no_vehicles = len(reward)
        for idx_1 in range(no_vehicles):
            self.agentmemory.memory_list[idx_1].rewards.append(reward[idx_1])
    

       


class AgentMemory:
    def __init__(self):
        self.memory_list = []
    def create_memory(self,amount):
        self.memory_list = []
        for _ in range(amount):
            self.memory_list.append(Memory())
    
    def delete_memory(self):
        del self.memory_list
    




def main():
    ############## Hyperparameters ##############
    env_name = "drones2d-v0"
    render = False                  # rendering mode, needs to be set to False.
    solved_reward = 100000          # stop training if avg_reward > solved_reward
    log_interval = 10               # print avg reward in the interval
    save_interval = 20              # Interval model is saved
    max_episodes = 100000           # max training episodes
    max_timesteps = 100             # max timesteps in one episode
    n_latent_var = [128,128,64]     # list of neurons in hidden layers
    update_timestep = 4000          # update policy every n timesteps
    action_std = 0.6                # constant std for action distribution
    lr = 0.0001                     # learning rate
    betas = (0.9, 0.999)            # betas
    gamma = 0.99                    # discount factor
    K_epochs = 2                    # update policy for K epochs
    eps_clip = 0.2                  # clip parameter for PPO
    random_seed = None              # random seed
    xrange_init = [-30,30]          # initial x-coordinate-range of vehicles
    yrange_init = [-30,30]          # initial y-coordinate-range of vehicles
    xrange_target = [-30,30]        # target x-coordinate-range of vehicles
    yrange_target = [-30,30]        # target y-coordinate-range of vehicles
    agents = 5                      # max no of agents in the simulation
    #############################################
    
    # creating environment
    env = gym.make(env_name)
    state_dim = 7
    action_dim = 2
    time_stamp = str(int(datetime.timestamp(datetime.now())))
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    
    memory = Memory()
    handler = AgentHandler()
    ppo = PPO(state_dim, action_dim, n_latent_var, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)
    file_string = "./logs/log_"+time_stamp
    f = open(file_string+"_parameters.txt","a+")
    f.write('Env-Name:\t\t{}\nn_latent_var:\t\t{}\nupdate_timestep:\t{}\naction_std:\t\t{}\nlr:\t\t\t{}\nbetas:\t\t\t{}\ngamma:\t\t\t{}\nK_epochs:\t\t{}\neps_clip:\t\t{}\nxrange_init:\t\t{}\nyrange_init:\t\t{}\nxrange_target:\t\t{}\nyrange_target:\t\t{}\n'.format(env_name,n_latent_var,update_timestep,action_std,lr,betas,gamma,K_epochs,eps_clip,xrange_init,yrange_init,xrange_target,yrange_target)) 
    f.close()

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        agents = np.random.randint(low=1,high=5)
        state,_ = env.reset(amount = agents,xrange_init=xrange_init,yrange_init=yrange_init,xrange_target=xrange_target,yrange_target=yrange_target,eps_arr=1)
        state = handler.state_handler(state)
        handler.agentmemory.delete_memory()     #delete old memory
        handler.agentmemory.create_memory(len(state))
        ##### Version with omitted position state #####
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = handler.select_action(state,ppo.policy_old.actor)       #for use with multi-agent environment
            response = env.step(action)
            # response = [response[0],response[1],response[2],response[4]]
            state,reward,done = handler.response_eval(response)

            # Saving reward:
            handler.reward_memory_append(reward)

            #Procedure to write the memory of done agents to the complete memory
            no_vehicles = len(done)
            del_list = []
            for idx_3 in range(no_vehicles):
                if done[idx_3]==True:
                    memory.states.extend(handler.agentmemory.memory_list[idx_3].states)
                    memory.actions.extend(handler.agentmemory.memory_list[idx_3].actions)
                    memory.rewards.extend(handler.agentmemory.memory_list[idx_3].rewards)
                    memory.logprobs.extend(handler.agentmemory.memory_list[idx_3].logprobs)
                    del_list.append(idx_3)
            
            del_list.sort(reverse=True)
            #delete done memory from agentmemory
            for idx_3b in range(len(del_list)):
                del handler.agentmemory.memory_list[del_list[idx_3b]]

            #if max_timesteps is reached copy agent's memory to central memory
            
            if t == max_timesteps-1:
                no_vehicles = len(handler.agentmemory.memory_list)

                for idx_4 in range(no_vehicles):
                    memory.states.extend(handler.agentmemory.memory_list[idx_4].states)
                    memory.actions.extend(handler.agentmemory.memory_list[idx_4].actions)
                    memory.rewards.extend(handler.agentmemory.memory_list[idx_4].rewards)
                    memory.logprobs.extend(handler.agentmemory.memory_list[idx_4].logprobs)

            running_reward += np.sum(reward)  #change to something more appliccaple
            if render:
                env.render()
            if all(elem == True for elem in done):    
                break
            response = handler.del_done(response)
            state,reward,done = handler.response_eval(response)
        
        #update after every n episode
        if i_episode % 10 == 0:
            ppo.update(memory)
            memory.clear_memory()
            time_step = 0

        avg_length += t
        
        #save the model at the last step
        if i_episode % save_interval == 0:
            print("Model saved!")
            torch.save(ppo.policy.state_dict(), './models/PPO_Continuous_{}_{}_{}.pth'.format(env_name,i_episode,time_stamp))
            #break

        #Log Avg_length and Avg_reward at every log_interval steps and write to file
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)+1
            running_reward = int((running_reward/log_interval/agents))
            
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            f = open(file_string+"_data.txt","a+")
            f.write('{}\t{}\t{}\n'.format(i_episode,avg_length,running_reward))
            f.close()
            running_reward = 0
            avg_length = 0
            
if __name__ == '__main__':
    main()
    