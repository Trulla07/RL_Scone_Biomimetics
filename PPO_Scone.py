import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta
from Environment import GaitGym2D
from sconetools import sconepy
from pdController import PD_Controller
from datetime import datetime
import copy
import os, shutil
import math
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GaussianActor_musigma(nn.Module):

    def __init__(self, state_dim, action_dim, net_width):
        super(GaussianActor_musigma, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.mu_head_muscles = nn.Linear(net_width, action_dim-4)
        self.sigma_head_muscles = nn.Linear(net_width, action_dim-4)
        self.mu_head_motor = nn.Linear(net_width,4)
        self.sigma_head_motor = nn.Linear(net_width,4)

    def forward(self,state):

        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        mu_muscles = torch.sigmoid(self.mu_head_muscles(a))
        sigma_muscles = F.softplus(self.sigma_head_muscles(a))

        b = torch.relu(self.l1(state))
        b = torch.relu(self.l2(b))
        mu_motors = torch.relu(self.mu_head_motor(b))
        sigma_motors = F.softplus(self.sigma_head_motor(b))
        if np.isnan(sigma_motors.detach().numpy()[0][0] ):
            print("Error")

        return mu_muscles, sigma_muscles, mu_motors, sigma_motors

    def get_dist(self, state):
        mu_muscles, sigma_muscles, mu_motors, sigma_motors = self.forward(state)
        dist_motors = Normal(mu_motors, sigma_motors)
        dist_muscles = Beta(mu_muscles, sigma_muscles)
        return dist_muscles, dist_motors

class Critic(nn.Module):

    def __init__(self, state_dim,net_width):
        super(Critic, self).__init__()

        self.C1 = nn.Linear(state_dim, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1) 

    def forward(self,state):
        v = torch.relu(self.C1(state))
        v = torch.relu(self.C2(v))
        v = self.C3(v)

        return v

class PPO(object):

    def __init__(self, state_dim, action_dim, env_with_Dead, path_save, gamma=0.99, lambd=0.95, clip_rate=0.2, K_epochs=10, net_width=64, a_lr=3e-4, c_lr=3e-4, l2_reg = 1e-3, a_optim_batch_size = 64,
		        c_optim_batch_size = 64, entropy_coef = 0, entropy_coef_decay = 0.9998):
        
        self.actor = GaussianActor_musigma(state_dim, action_dim, net_width).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

        self.critic = Critic(state_dim, net_width).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

        self.env_with_Dead = env_with_Dead
        self.action_dim = action_dim
        self.clip_rate = clip_rate
        self.gamma = gamma
        self.lambd = lambd
        self.clip_rate = clip_rate
        self.K_epochs = K_epochs
        self.l2_reg = l2_reg
        self.a_optim_batch_size = a_optim_batch_size
        self.c_optim_batch_size = c_optim_batch_size
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.path_save = path_save
        self.data = []

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            dist_muscles, dist_motors = self.actor.get_dist(state)
            a = dist_muscles.sample()
            a = torch.clamp(a, 0, 1)
            logprob_a_muscles = dist_muscles.log_prob(a).cpu().detach().numpy().flatten()

            b = dist_motors.sample()
            logprob_b_motors = dist_motors.log_prob(b).cpu().detach().numpy().flatten()
                
            return a.cpu().numpy().flatten(), logprob_a_muscles, b.cpu().numpy().flatten(), logprob_b_motors

    def evaluate(self,state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            mu_muscles, sigma_muscles, mu_motors, sigma_motors = self.actor(state)
                
            return mu_muscles.cpu().detach().numpy().flatten(),0.0, mu_motors.cpu().detach().numpy().flatten(),0.0 

    def train(self):
        
        self.entropy_coef*=self.entropy_coef_decay
        s, a_muscles, a_motor, r, s_prime, logprob_a_muscles, logprob_motor, done_mask, dw_mask = self.make_batch()
        
        ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_prime)

            '''dw for TD_target and Adv'''
            deltas = r + self.gamma * vs_ * (1 - dw_mask) - vs

            deltas = deltas.cpu().flatten().numpy()
            adv = [0]

            '''done for GAE'''
            for dlt, mask in zip(deltas[::-1], done_mask.cpu().flatten().numpy()[::-1]):
                advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - mask)
                adv.append(advantage)
            adv.reverse()
            adv = copy.deepcopy(adv[0:-1])
            adv = torch.tensor(adv).unsqueeze(1).float().to(device)
            td_target = adv + vs
            adv = (adv - adv.mean()) / ((adv.std()+1e-4))  #sometimes helps


        """Slice long trajectopy into short trajectory and perform mini-batch PPO update"""
        a_optim_iter_num = int(math.ceil(s.shape[0] / self.a_optim_batch_size))
        c_optim_iter_num = int(math.ceil(s.shape[0] / self.c_optim_batch_size))
        for i in range(self.K_epochs):

            #Shuffle the trajectory, Good for training
            perm = np.arange(s.shape[0])
            np.random.shuffle(perm)
            perm = torch.LongTensor(perm).to(device)
            s, a_muscles, a_motor, td_target, adv, logprob_a_muscles, logprob_motor = \
				s[perm].clone(), a_muscles[perm].clone(), a_motor[perm].clone(),  td_target[perm].clone(), adv[perm].clone(), logprob_a_muscles[perm].clone(), logprob_motor[perm].clone()

            '''update the actor'''
            for i in range(a_optim_iter_num):
                index = slice(i * self.a_optim_batch_size, min((i + 1) * self.a_optim_batch_size, s.shape[0]))
                distribution_muscles, dist_motors = self.actor.get_dist(s[index])
                dist_entropy_muscles = distribution_muscles.entropy().sum(1, keepdim=True)
                logprob_a_now = distribution_muscles.log_prob(a_muscles[index])
                ratio = torch.exp(logprob_a_now.sum(1,keepdim=True) - logprob_a_muscles[index].sum(1,keepdim=True))  # a/b == exp(log(a)-log(b))


                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, min = 1 - self.clip_rate, max = 1 + self.clip_rate) * adv[index]
                a_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy_muscles

                dist_entropy_motors = dist_motors.entropy().sum(1,keepdim=True)
                logprob_motor_now = dist_motors.log_prob(a_motor[index])
                ratio_motor = torch.exp(logprob_motor_now.sum(1,keepdim = True)-logprob_motor[index].sum(1,keepdim=True))
                surr1 = ratio_motor*adv[index]
                surr2 = torch.clamp(ratio_motor, min=1-self.clip_rate, max=1+self.clip_rate)*adv[index]
                a_loss2 = -torch.min(surr1,surr2) - self.entropy_coef* dist_entropy_motors

                loss = a_loss + a_loss2

                self.actor_optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                self.actor_optimizer.step()

            '''update the critic'''
            for i in range(c_optim_iter_num):
                index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, s.shape[0]))
                c_loss = (self.critic(s[index]) - td_target[index]).pow(2).mean()
                for name,param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                self.critic_optimizer.step()

    def make_batch(self):

        s_lst, a_muscles_lst, a_motor_lst, r_lst, s_prime_lst, logprob_a_muscles_lst, logprob_a_motor_lst, done_lst, dw_lst = [], [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a_muscles, a_motor, r, s_prime, logprob_a_muscles, logprob_a_motor, done, dw = transition

            s_lst.append(s)
            a_muscles_lst.append(a_muscles)
            a_motor_lst.append(a_motor)
            logprob_a_muscles_lst.append(logprob_a_muscles)
            logprob_a_motor_lst.append(logprob_a_motor)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([done])
            dw_lst.append([dw])

        if not self.env_with_Dead:
            '''Important!!!'''
			# env_without_DeadAndWin: deltas = r + self.gamma * vs_ - vs
			# env_with_DeadAndWin: deltas = r + self.gamma * vs_ * (1 - dw) - vs
            dw_lst = (np.array(dw_lst)*False).tolist()

        self.data = [] #Clean history trajectory

        '''list to tensor'''
        with torch.no_grad():
            s, a_muscles, a_motor, r, s_prime, logprob_a_muscles, logprob_a_motor, done_mask, dw_mask = \
				torch.tensor(s_lst, dtype=torch.float).to(device), \
				torch.tensor(a_muscles_lst, dtype=torch.float).to(device), \
                torch.tensor(a_motor_lst, dtype=torch.float).to(device), \
				torch.tensor(r_lst, dtype=torch.float).to(device), \
				torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
				torch.tensor(logprob_a_muscles_lst, dtype=torch.float).to(device), \
                torch.tensor(logprob_a_motor_lst, dtype=torch.float).to(device), \
				torch.tensor(done_lst, dtype=torch.float).to(device), \
				torch.tensor(dw_lst, dtype=torch.float).to(device),


        return s, a_muscles, a_motor, r, s_prime, logprob_a_muscles, logprob_a_motor, done_mask, dw_mask

    def put_data(self, transition):
        self.data.append(transition)

    def save(self,episode,score):
        torch.save(self.critic.state_dict(),f"./model/{self.path_save}/ppo_critic{episode}_{score}.pth")
        torch.save(self.actor.state_dict(), f"./model/{self.path_save}/ppo_actor{episode}_{score}.pth".format(episode,score))


    def load(self,episode,score):
        self.critic.load_state_dict(torch.load("./model/results/new/0.0002_0.0002_0.99_0.95_0.2_10_150_2048_50000000.0_5000.0_64_64_0.001_0.99/ppo_critic{}_{}.pth".format(episode, score)))
        self.actor.load_state_dict(torch.load("./model/results/new/0.0002_0.0002_0.99_0.95_0.2_10_150_2048_50000000.0_5000.0_64_64_0.001_0.99/ppo_actor{}_{}.pth".format(episode,score)))

def evaluate_policy(env, model, render, max_steps, max_action, pd_controller):
    scores = 0
    turns = 3
    for j in range(turns):

        s, done, ep_r, steps = env.reset(), False, 0, 0
        while not (done or (steps >= max_steps)):
            # Take deterministic actions at test time
            a_muscles, logprob_a, a_motor, logprob_a_motor = model.evaluate(s)

            action_hip_r, action_knee_r, action_hip_l, action_knee_l = pd_controller.pd_control(s, a_motor[0], a_motor[1])

            if np.isnan(action_hip_r) or np.isnan(action_knee_r) or np.isnan(action_hip_l) or  np.isnan(action_knee_l):
                print("Error")

            if np.abs(action_hip_r) > 2:
                action_hip_r = 0
            if np.abs(action_knee_r) >2:
                action_knee_r = 0
            if np.abs(action_hip_l) > 2:
                action_hip_l = 0
            if np.abs(action_knee_l) >2:
                action_knee_l = 0


            action_hip_r = np.expand_dims(action_hip_r,axis = 0)
            action_knee_r = np.expand_dims(action_knee_r,axis = 0)
            action_hip_l = np.expand_dims(action_hip_l,axis = 0)
            action_knee_l = np.expand_dims(action_knee_l,axis = 0)

            act = np.concatenate([a_muscles, action_hip_r, action_knee_r, action_hip_l, action_knee_l])

            s_prime, reward, done, info = env.step(act)

            ep_r += reward
            steps += 1
            s = s_prime
            if render:
                env.render()
        scores += ep_r
    return scores/turns

def main(EnvIdex, write, render, Loadmodel, ModelIdex, ModelScore, seed, T_horizon, distnum, Max_train_steps, save_interval, eval_interval, 
         gamma, lambd, clip_rate, K_epochs, net_width, a_lr, c_lr, l2_reg, a_optim_batch_size, c_optim_batch_size, entropy_coef, entropy_coef_decay):

    path_save = "results/new/%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" %(a_lr, c_lr, gamma, lambd, clip_rate, K_epochs, net_width,T_horizon, Max_train_steps,eval_interval, a_optim_batch_size, c_optim_batch_size, entropy_coef, entropy_coef_decay)
    write = write   #Use SummaryWriter to record the training.
    render = render

    avg_rewards = []
    avg_score = []

    #state_obj = State()
    best_score = 0
    bsafe = False
    env_with_Dead = True
    env  = GaitGym2D()
    eval_env = GaitGym2D()
    pd_controller = PD_Controller()
    state_dim = 90
    action_dim = 20#env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_steps = 10000 # 10 sec á 0.001 sec steps
    print('Env: scone-rl','  state_dim:',state_dim,'  action_dim:',action_dim,
            '  max_a:',max_action,'  min_a:',env.action_space.low[0])
    T_horizon = T_horizon  #lenth of long trajectory

    Max_train_steps = Max_train_steps
    save_interval = save_interval#in steps
    eval_interval = eval_interval#in steps

    # random_seed = seed
    # print("Random Seed: {}".format(random_seed))
    #torch.manual_seed(random_seed)
    #env.seed(random_seed)
    # eval_env.seed(random_seed)
    # np.random.seed(random_seed)

    if write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'+timenow
        if os.path.exists(writepath): 
            shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "env_with_Dead":env_with_Dead,
        "gamma": gamma,
        "lambd": lambd,     #For GAE
        "clip_rate": clip_rate,  #0.2
        "K_epochs": K_epochs,
        "net_width": net_width,
        "a_lr": a_lr,
        "c_lr": c_lr,
        "l2_reg": l2_reg,   #L2 regulization for Critic
        "a_optim_batch_size":a_optim_batch_size,
        "c_optim_batch_size": c_optim_batch_size,
        "entropy_coef":entropy_coef, #Entropy Loss for Actor: Large entropy_coef for large exploration, but is harm for convergence.
        "entropy_coef_decay":entropy_coef_decay,
        "path_save":path_save
    }

    if not os.path.exists('model'): 
        os.mkdir('model')

    model = PPO(**kwargs)

    if Loadmodel: 
        model.load(ModelIdex, ModelScore)

    traj_lenth = 0
    total_steps = 0
    while total_steps < Max_train_steps:

        s, done, steps, ep_r = env.reset(), False, 0, 0

        #s = state_obj.convertToNumpy(s)

        '''Interact & train'''
        while not done:
            traj_lenth += 1
            steps += 1

            if render:
                env.render()
                a_muscles, logprob_a_muscles, a_motor, logprob_a_motor = model.evaluate(s)
            else:
                a_muscles, logprob_a_muscles, a_motor, logprob_a_motor = model.select_action(s)
            
            action_hip_r, action_knee_r, action_hip_l, action_knee_l = pd_controller.pd_control(s, a_motor[0], a_motor[1])
            if np.abs(action_hip_r) > 2:
                action_hip_r = 0
            if np.abs(action_knee_r) >2:
                action_knee_r = 0
            if np.abs(action_hip_l) > 2:
                action_hip_l = 0
            if np.abs(action_knee_l) >2:
                action_knee_l = 0

            action_hip_r = np.expand_dims(action_hip_r,axis = 0)
            action_knee_r = np.expand_dims(action_knee_r,axis = 0)
            action_hip_l = np.expand_dims(action_hip_l,axis = 0)
            action_knee_l = np.expand_dims(action_knee_l,axis = 0)

            act = np.concatenate([a_muscles, action_hip_r, action_knee_r, action_hip_l, action_knee_l])
            s_prime, reward, done, info = env.step(act)

            if steps==max_steps:
                done = True

            '''distinguish done between dead|win(dw) and reach env._max_episode_steps(rmax); done = dead|win|rmax'''
            '''dw for TD_target and Adv; done for GAE'''
            if done and steps != max_steps: 
                dw = True
                #still have exception: dead or win at _max_episode_steps will not be regard as dw.
                #Thus, decide dw according to reward signal of each game is better.  dw = done_adapter(r)
            else:
                dw = False

            model.put_data((s, a_muscles, a_motor, reward, s_prime, logprob_a_muscles, logprob_a_motor, done, dw))
            s = s_prime
            ep_r += reward

            '''update if its time'''
            if not render:
                if traj_lenth % T_horizon == 0:
                    model.train()
                    traj_lenth = 0

            '''record & log'''
            if total_steps % eval_interval == 0:
                score = evaluate_policy(eval_env, model, False, max_steps, max_action, pd_controller)
                if write:
                    writer.add_scalar('ep_r_insteps', score, global_step=total_steps)
                # print('EnvName: scone','steps: {}k'.format(int(total_steps/1000)),'score:', score)
                print('EnvName: opensim','steps: {}k'.format(int(total_steps/1000)),'score:', score, 'steps:', steps)
                avg_score.append(score)
                avg_rewards.append(ep_r/steps)
                if not os.path.isdir(f"./model/{path_save}"):
                    os.makedirs(f"./model/{path_save}")
                np.save(f"./model/{path_save}/avgRewards.npy", avg_rewards)
                np.save(f"./model/{path_save}/avgScore.npy", avg_score)
                if best_score < score:
                    best_score = score
                    model.save(total_steps, np.round(score, decimals = 1))

            total_steps += 1

            if total_steps % save_interval==0:
                bsafe = True

            '''save model'''
        if bsafe:
            env.store_next_episode()
            bsafe=False

    env.close()

    return True


if __name__ == '__main__':

    EnvIdex = 0 
    write = True # Use summaryWriter to record training
    render = False # render or not
    Loadmodel = False # load old model or not
    ModelIdex = 345000 # which model to load
    ModelScore =231.1 # which model to load
    seed = 0 # random seed
    T_horizon = 2048 # length of long trajectory
    distnum = 1 #0:Beta ; 1:GS_ms  ;  2: GS_m
    Max_train_steps = 5e7 # max train steps
    save_interval = 10000 # model saving intervals in steps
    eval_interval = 5e3 # model evaluaing interval in steps
    gamma = 0.99 # discount factor
    lambd = 0.95 # GAE Factor
    clip_rate = 0.2 # ppo clip rate
    K_epochs = 10 # ppo update times
    net_width = 64 # hidden net width
    a_lr = 2e-4 # learning rate of actor
    c_lr = 2e-4 # learning rate of critic
    l2_reg = 1e-3 # L2 regulization coefficient for critic
    a_optim_batch_size = 64 # length of sliced trajectory of actor
    c_optim_batch_size = 64 # length of sliced trajectory of critic
    entropy_coef = 1e-3 # entropy coefficient of actor
    entropy_coef_decay = 0.99 # decay rate of entropy_coef

    main(EnvIdex, write, render, Loadmodel, ModelIdex, ModelScore, seed, T_horizon, distnum, Max_train_steps, save_interval, eval_interval, 
        gamma, lambd, clip_rate, K_epochs, net_width, a_lr, c_lr, l2_reg, a_optim_batch_size, c_optim_batch_size, entropy_coef, entropy_coef_decay)
    
    # env = GaitGym2D()

    # for episode in range(100):

    #     if episode%10 == 0:
    #         env.store_next_episode()

    #     episode_steps = 0
    #     total_reward = 0
    #     state = env.reset()

    #     while True:
    #         action = env.action_space.sample()
    #         next_state, reward, done, info = env.step(action)
    #         episode_steps+=1
    #         total_reward+=reward

    #         if done or (episode_steps>=100):
    #             print(f'Episode {episode} finsihed. Steps = {episode_steps}, reward = {total_reward:0.3f}')
    #             break

