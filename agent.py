import torch, numpy as np
from tensordict.tensordict import TensorDict
from torchrl.data import (
    TensorDictReplayBuffer,
    LazyMemmapStorage
)

from constants import device
from neural import MarioNet


class Mario:
    def __init__(self, inputs, actions, save_dir, checkpoint=None):
        self.inputs = inputs
        self.actions = actions
        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(
                max_size=100_000,
                device=torch.device("cpu")
            )
        )
        self.batch_size = 32

        self.explored = 0
        self.exploration_rate = 1
        self.exploration_rate_decay = .9999995  # 99999975
        self.exploration_rate_min = .1
        self.gamma = .9

        self.curr_step = 0
        self.not_learn_before = 20_00  # 100000
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 20_00  # no. of experiences between Q_target & Q_online sync (1e4)

        self.save_every = 1e4  # no. of experiences between saving Mario Net (5e5)
        self.save_dir = save_dir

        self.device = device
        # Mario's DNN to predict the most optimal action
        self.net = MarioNet(self.inputs, self.actions).float()
        self.check_model_parameters_device()

        if checkpoint:
            self.load(checkpoint)

        # from author: Adam
        # we must try others
        # (find out (from math, field of task),
        # why they can be better)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=.00025
        )
        # from author: SmoothL1Loss
        self.loss_fn = torch.nn.SmoothL1Loss()  # MSELoss

    def act(self, state):
        """Given a state, choose an epsilon-greedy action and update value of step.
            Inputs: state(LazyFrame): A single observation of the current state, dimension is (state_dim)
            Outputs: action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.actions)

        # EXPLOIT
        else:
            self.explored += 1
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(
                data=state,
                dtype=torch.float32,  # error: MPS supports only float32
                device=self.device
            ).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        if self.exploration_rate < self.exploration_rate_min:
            self.exploration_rate = self.exploration_rate_min
        else:
            self.exploration_rate *= self.exploration_rate_decay

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """Store the experience to self.memory (replay buffer)
            Inputs:
                state (``LazyFrame``),
                next_state (``LazyFrame``),
                action (``int``),
                reward (``float``),
                done(``bool``)
        """

        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        self.memory.add(
            TensorDict(
                source={
                    "state": state,
                    "next_state": next_state,
                    "action": action,
                    "reward": reward,
                    "done": done
                },
                batch_size=[]
            )
        )

    def recall(self):
        """
        Mario randomly samples a batch of experiences
        from his memory, and uses that to learn the game.
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        # if u have problems with torchrl, u can install:
        # batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = (
            batch.get(key) for key in
            ("state", "next_state", "action", "reward", "done"))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model='online')[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, terminated):
        next_state_Q = self.net(next_state, model='online')
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model='target')[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - terminated.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(
            self.net.online.state_dict()
        )

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.not_learn_before:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return td_est.mean().item(), loss

    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(
            load_path,
            map_location=device
        )
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(
            f"Loading model at {load_path} with"
            f"exploration rate {exploration_rate}"
        )
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate

    def check_model_parameters_device(self):
        print('Parameters of model is created on: ', end='')
        first_param_device = next(self.net.parameters()).device
        for parameter in self.net.parameters():
            if parameter.device != first_param_device:
                print(parameter.device)

        print(first_param_device)
