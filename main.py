import pygame as pg
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from collections import deque
import constants as cs
import pickle
import os
import sys

pg.init()

class Vacuum(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Env(object):
    def __init__(self):
        self.screen = pg.display.set_mode(cs.win_dims)
        pg.display.set_caption("Vacuum Cleaner")

        self.grid_width = cs.win_width // cs.cell_size
        self.grid_height = cs.win_height // cs.cell_size

        self.vac = Vacuum(self.grid_width // 2, self.grid_height // 2)
        self.trash = []

    def get_state(self):
        state = []
        for y in range(self.vac.y - cs.vision_x, self.vac.y + cs.vision_y + 1):
            for x in range(self.vac.x - cs.vision_x, self.vac.x + cs.vision_y + 1):
                if x < 0 or y < 0 or x >= self.grid_width or y >= self.grid_height:
                    state.append(0)
                elif (x, y) in self.trash:
                    state.append(1)
                else:
                    state.append(0.5)

        return state

    def distribute_trash(self):
        self.trash = []

        for _ in range(70):
            x = np.random.randint(1, self.grid_width - 1)
            y = np.random.randint(1, self.grid_height - 1)

            while (x, y) in self.trash or (x, y) == (self.vac.x, self.vac.y):
                x = np.random.randint(1, self.grid_width - 1)
                y = np.random.randint(1, self.grid_height - 1)
            
            self.trash.append((x, y))

    def get_reward(self, action):
        x = self.vac.x
        y = self.vac.y

        outta_bounds = (action == 0 and y == 0) or \
            (action == 1 and x == self.grid_width - 1) or \
                (action == 2 and y == self.grid_height - 1) or \
                    (action == 3 and x == 0)

        if outta_bounds:
            return -100

        if (x, y) in self.trash:
            self.trash.remove((x, y))
            if len(self.trash) == 0:
                self.distribute_trash()
            return 50

        else:
            return 0

    def step(self, action):
        if action == 0 and self.vac.y > 0:
            self.vac.y -= 1
        elif action == 1 and self.vac.x < self.grid_width - 1:
            self.vac.x += 1
        elif action == 2 and self.vac.y < self.grid_height - 1:
            self.vac.y += 1
        elif action == 3 and self.vac.x > 0:
            self.vac.x -= 1

        reward = self.get_reward(action)
        new_state = self.get_state()
        terminate = False if reward >= 0 else True

        return new_state, reward, terminate

    def reset(self):
        self.vac = Vacuum(self.grid_width // 2, self.grid_height // 2)
        self.distribute_trash()
        state = self.get_state()

        return state

    def render(self):
        self.screen.fill((250, 250, 250))
        for y in range(self.vac.y - cs.vision_x, self.vac.y + cs.vision_y + 1):
            for x in range(self.vac.x - cs.vision_x, self.vac.x + cs.vision_y + 1):
                if x >= 0 and x < self.grid_width and y >= 0 and y < self.grid_height:
                    pg.draw.rect(self.screen,
                                 (180, 180, 180),
                                 pg.Rect(x * cs.cell_size, y * cs.cell_size, cs.cell_size, cs.cell_size))

        pg.draw.rect(self.screen,
                     (128, 0, 128),
                     pg.Rect(self.vac.x * cs.cell_size, self.vac.y * cs.cell_size, cs.cell_size, cs.cell_size))

        for (x, y) in self.trash:
            pg.draw.rect(self.screen,
                         (139, 69, 19),
                         (x * cs.cell_size, y * cs.cell_size, cs.cell_size, cs.cell_size))

        for x in range(self.grid_width):
            pg.draw.line(self.screen, (0, 0, 0), (x * cs.cell_size, 0), (x * cs.cell_size, cs.win_height))

        for y in range(self.grid_height):
            pg.draw.line(self.screen, (0, 0, 0), (0, y * cs.cell_size), (cs.win_width, y * cs.cell_size))

        pg.display.flip()


class ReplayBuffer(object):
    def __init__(self):
        self.states = deque(maxlen=cs.replay_buffer_len)
        self.actions = deque(maxlen=cs.replay_buffer_len)
        self.rewards = deque(maxlen=cs.replay_buffer_len)
        self.states_ = deque(maxlen=cs.replay_buffer_len)
        self.terminals = deque(maxlen=cs.replay_buffer_len)
        self.len = 0

    def append(self, state, action, reward, state_, terminal):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.states_.append(state_)
        self.terminals.append(0 if terminal else 1)
        self.len += 1

    def get_random_samples(self):
        if self.len < cs.batch_size: return

        indices = np.random.randint(self.len % cs.replay_buffer_len, size=cs.batch_size)

        states = np.array([self.states[i] for i in indices])
        actions = np.array([self.actions[i] for i in indices])
        rewards = np.array([self.rewards[i] for i in indices])
        states_ = np.array([self.states_[i] for i in indices])
        terminals = np.array([self.terminals[i] for i in indices])

        return states, actions, rewards, states_, terminals


class DQNAgent(object):
    def __init__(self, env):
        self.env = env

    def new_model(self):
        model = keras.models.Sequential([
            keras.layers.Dense(cs.hidden_1, input_shape=(cs.inp_size,)),
            keras.layers.Activation(cs.hidden_1_activation),
            keras.layers.Dense(cs.hidden_2),
            keras.layers.Activation(cs.hidden_2_activation),
            keras.layers.Dense(cs.output)
        ])

        model.compile(optimizer=keras.optimizers.Adam(lr=cs.learning_rate), loss='mse')

        self.q_eval = model
        self.q_target = keras.models.clone_model(self.q_eval)
        self.epsilon = cs.epsilon
        self.rb = ReplayBuffer()

    def save_model(self):
        self.q_eval.save(cs.model_file, overwrite=True)
        with open(cs.rb_file, 'wb') as rb_file:
            pickle.dump(self.rb, rb_file)

    def load_model(self):
        self.q_eval = keras.models.load_model(cs.model_file)
        self.q_target = keras.models.load_model(cs.model_file)

        with open(cs.rb_file, 'rb') as rb_file:
            self.rb = pickle.load(rb_file)

        self.epsilon = 0.1

    def epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(cs.output)
        else:
            q_values = self.q_eval.predict(np.array(state)[np.newaxis])
            return np.argmax(q_values)

    def step(self, state):
        action = self.epsilon_greedy(state)
        state_, reward, terminal = self.env.step(action)
        self.rb.append(state, action, reward, state_, terminal)
        return state_, reward, terminal

    def reinforce(self):
        if self.rb.len < cs.batch_size: return

        states, actions, rewards, states_, terminals = self.rb.get_random_samples()
        
        next_q = self.q_target.predict(states_)
        next_eval = self.q_eval.predict(states_)
        argmax = np.argmax(next_eval, axis=1)
        target_q = self.q_eval.predict(states)

        idx = np.arange(cs.batch_size, dtype=np.int32)

        target_q[idx, actions] = rewards + cs.gamma * \
            next_q[idx, argmax.astype(int)] * terminals

        self.q_eval.fit(states, target_q, verbose=0)
        self.epsilon = max(self.epsilon * cs.epsilon_dec, 0.01)

    def commence(self, mode):
        steps = 0
        target_weight_switch = 100
        training = mode != 'e'
        to_render = True

        while True:
            ep_done = False
            obs = self.env.reset()

            for _ in range(500):
                obs, reward, terminal = self.step(obs)
                steps += 1
                ep_done = terminal

                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        if training:
                            self.save_model()
                        pg.quit()
                        quit()
                    
                    if event.type == pg.KEYDOWN:
                        if event.key == pg.K_r:
                            to_render = not to_render

                if to_render:
                    self.env.render()

                if training:
                    self.reinforce()

                    if steps % target_weight_switch == 0:
                        self.q_target.set_weights(self.q_eval.get_weights())

                if ep_done:
                    break



def main():
    if len(sys.argv) != 2:
        print(cs.usage)
        return

    if sys.argv[1] not in ['n', 'c', 'e']:
        print(cs.usage)
        return

    mode = sys.argv[1]

    agent = DQNAgent(Env())

    if mode == 'n':
        agent.new_model()
    else:
        agent.load_model()

    agent.commence(mode)


if __name__ == '__main__':
    main()
