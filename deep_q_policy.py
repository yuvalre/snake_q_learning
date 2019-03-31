from policies import base_policy as bp
import numpy as np
import random
from copy import deepcopy
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

LEARNING_TIME = bp.LEARNING_TIME

NUM_SYMBOLS = 11
DIRECTIONS = ('N', 'W', 'E', 'S')

# Default parameters:
BATCH_SIZE = 16
EPSILON = 0.5
GAMMA = 0.7
LEARNING_RATE = 0.0001
SPATIAL_BOX_RADIUS = 5
DISTANCE_SCOPE_RADIUS = 15
REPLAY_SIZE = 5000
DROPOUT = 0.3

# Parameters for linear policy takeover at the beginning of the game
LINEAR_TAKEOVER_DURATION = 0
LINEAR_GAMMA = 0.5
LINEAR_LR = 0.01
NUM_ACTIONS = 3
STEPS_AHEAD = 2

EPSILON_DECAY_DURATION_PERCENT = 0.2


class Custom308490234(bp.Policy):
    """
    A deep Q-Learning approximation which avoids collisions with obstacles and other snakes and
    wishes to eat good yummy fruits.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(
            policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['gamma'] = float(
            policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        policy_args['lr'] = float(
            policy_args['lr']) if 'lr' in policy_args else LEARNING_RATE
        policy_args['bs'] = int(
            policy_args['bs']) if 'bs' in policy_args else BATCH_SIZE
        policy_args['spatial_r'] = int(
            policy_args['spatial_r']) if 'spatial_r' in policy_args else SPATIAL_BOX_RADIUS
        policy_args['dist_r'] = int(
            policy_args['dist_r']) if 'dist_r' in policy_args else DISTANCE_SCOPE_RADIUS
        policy_args['max_replay'] = int(
            policy_args['max_replay']) if 'max_replay' in policy_args else REPLAY_SIZE
        policy_args['dropout'] = int(
            policy_args['dropout']) if 'dropout' in policy_args else DROPOUT
        policy_args['ltd'] = int(
            policy_args['ltd']) if 'ltd' in policy_args else LINEAR_TAKEOVER_DURATION
        policy_args['edd'] = float(
            policy_args['edd']) if 'edd' in policy_args else EPSILON_DECAY_DURATION_PERCENT

        return policy_args

    def init_run(self):
        self.r_sum = 0

        # Give each action an Index to use in the replay history
        self.act2idx = {'L': 0, 'R': 1, 'F': 2}
        self.idx2act = {0: 'L', 1: 'R', 2: 'F'}

        # Feature parameters
        #####################
        self.rotate_num = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        # Distance:
        self.distance_scope = min(self.dist_r, min(self.board_size[0], self.board_size[1]) // 2)
        # Spatial:
        self.spatial_box = min(self.spatial_r, (min(self.board_size[0], self.board_size[1]) - 1) // 2)
        box_feature_num = (((self.spatial_box * 2) + 1) ** 2) * NUM_SYMBOLS

        # Overall:
        self.n_features = ((self.distance_scope + 1) * NUM_SYMBOLS) + box_feature_num

        # Replay buffer
        ################
        self.replay_prev = np.zeros(shape=(self.max_replay, self.n_features))
        self.replay_next = np.zeros(shape=(self.max_replay, 3, self.n_features))
        self.replay_reward = np.zeros(shape=(self.max_replay))
        self.replay_action = np.zeros(shape=(self.max_replay), dtype=np.int8)
        self.replay_idx = 0

        # A flag for the first round
        self.first_act = True

        # Initialize parameters necessary for the distance & spatial features
        self.init_roll_params()

        # Linear Takeover
        ##################
        # Run a linear model for the first k rounds
        self.linear_takeover_on = True
        # Parameters for the linear policy:
        self.linear_lr = LINEAR_LR
        self.linear_gamma = LINEAR_GAMMA
        self.linear_n_features = NUM_SYMBOLS + ((STEPS_AHEAD - 1) * NUM_ACTIONS * NUM_SYMBOLS) + 1
        self.linear_buffi = 0
        self.linear_feature_buff = np.zeros(shape=(LEARNING_TIME * 2, self.linear_n_features))
        self.linear_deltas = np.zeros(shape=(LEARNING_TIME * 2, 1))
        self.linear_w = np.zeros(self.linear_n_features)

        # Epsilon-Decay Parameters
        ###########################
        self.linear_epsilon_decay_scope = self.edd * self.game_duration
        self.linear_epsilon_decay_start = self.game_duration - \
                                          self.score_scope - self.linear_epsilon_decay_scope
        if self.linear_epsilon_decay_scope:
            self.linear_epsilon_decay = self.epsilon/self.linear_epsilon_decay_scope
        else:
            self.linear_epsilon_decay = 0

        self.init_net()


    def linear_get_q(self, state, action):
        """
        Returns the feature vector f(s,a) and current estimation of Q(s,a).
        The feature vector is a one-hot encoding vector of the values of the next 2 steps:
        - the value at the next position (where we'll be after the specified action)
        - and the values of the 3 possible next-next-turn positions after that action.
        """
        features = np.zeros(self.linear_n_features)

        board, head = state
        head_pos, direction = head

        # first step:
        next_direction = bp.Policy.TURNS[direction][action]
        next_position = head_pos.move(next_direction)
        r = next_position[0]
        c = next_position[1]
        features[board[r, c] + 1] = 1
        indices = [board[r, c] + 1]

        # second step:
        for i, a in enumerate(bp.Policy.ACTIONS):
            a_direction = bp.Policy.TURNS[next_direction][a]
            a_position = next_position.move(a_direction)
            a_r, a_c = a_position[0], a_position[1]
            f_index = ((i + 1) * NUM_SYMBOLS) + board[a_r, a_c] + 1
            features[f_index] = 1
            indices.append(f_index)

        # bias:
        features[-1] = 1
        indices.append(self.linear_n_features-1)

        return features, self.linear_w[indices].sum()

    def linear_act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        the function for choosing an action, given current state, *during the linear takeover*.
        it accepts the state-action-reward needed to learn from the previous
        move (which it can save in a data structure for future learning), and
        accepts the new state from which it needs to decide how to act.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
        :param too_slow: true if the game didn't get an action in time from the
                        policy for a few rounds in a row. use this to make your
                        computation time smaller (by lowering the batch size for example)...
        :return: an action (from Policy.Actions) in response to the new_state.
        """
        board, head = new_state
        head_pos, direction = head

        prev_features, prev_q = self.linear_get_q(prev_state, prev_action)
        self.linear_feature_buff[self.linear_buffi] = prev_features

        # calculate delta and best act:
        qs = []
        acts = list(np.random.permutation(bp.Policy.ACTIONS))
        for a in acts:
            features, q = self.linear_get_q(new_state, a)
            qs.append(q)
        best_act = acts[np.argmax(qs)]
        best_q = np.max(qs)

        delta = prev_q - (reward + (self.linear_gamma * best_q))
        self.linear_deltas[self.linear_buffi] = delta
        self.linear_buffi += 1

        if round >= self.ltd:  # check for linear takeover duration
            self.linear_takeover_on = False

        return best_act

    def linear_learn(self):
        """
        the function for learning and improving the linear policy *during the linear takeover*.
        it accepts the state-action-reward needed to learn from the final move of the game,
        and from that (and other state-action-rewards saved previously) it
        improves the policy.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
                          This is the final state of the round.
        :param too_slow: true if the game didn't get an action in time from the
                        policy for a few rounds in a row. you may use this to make your
                        computation time smaller (by lowering the batch size for example).
        """
        max_buffi = max(self.linear_buffi, 5)
        self.linear_deltas = self.linear_deltas[:max_buffi]
        self.linear_feature_buff = self.linear_feature_buff[:max_buffi]
        self.linear_buffi = 0
        self.linear_w = self.linear_w - self.linear_lr * (self.linear_deltas * self.linear_feature_buff).mean(axis=0)

    def init_roll_params(self):
        """
        Initializes parameters necessary for extracting the distance & spatial features.
        Creates maps for mapping between the snake's direction and the masks that are applied on the board.
        A fuller explanation of the reasoning behind these maps can be found at Answers.pdf.
        """
        self.roll_params = dict()
        self.radius_map = dict()

        h, w = self.board_size

        # "The center of the board" needs to be defined if one of the board's dimensions is even.
        hight_even = 1 - int(h % 2)
        width_even = 1 - int(w % 2)
        r = h // 2  # center row index
        c = w // 2  # center column index

        # North
        above = r
        below = r - hight_even
        left = c
        right = c - width_even

        north = {'r': r, 'c': c, 'above': above, 'below': below, 'left': left, 'right': right,
                 'box': (r - 2, r + 2, c)}
        self.roll_params['N'] = north

        self.radius_map['N'] = dict()
        self.radius_map['N']['r'] = np.arange(-1 * north['above'], north['below'] + 1)
        self.radius_map['N']['c'] = np.arange(-1 * north['left'], north['right'] + 1)

        # South
        # if h is odd so same as north, else every row should roll up by 1
        south = deepcopy(north)
        south['r'] -= hight_even
        south['above'] -= hight_even
        south['below'] += hight_even
        self.roll_params['S'] = south

        self.radius_map['S'] = dict()
        self.radius_map['S']['r'] = np.arange(-1 * south['above'], south['below'] + 1)
        self.radius_map['S']['c'] = np.arange(-1 * south['left'], south['right'] + 1)

        # West
        west = deepcopy(north)
        self.roll_params['W'] = west

        self.radius_map['W'] = dict()
        self.radius_map['W']['r'] = np.arange(-1 * west['above'], west['below'] + 1)
        self.radius_map['W']['c'] = np.arange(-1 * west['left'], west['right'] + 1)

        # East
        east = deepcopy(west)
        east['c'] -= width_even
        east['left'] -= width_even
        east['right'] += width_even
        self.roll_params['E'] = east

        self.radius_map['E'] = dict()
        self.radius_map['E']['r'] = np.arange(-1 * east['above'], east['below'] + 1)
        self.radius_map['E']['c'] = np.arange(-1 * east['left'], east['right'] + 1)

        for dir in self.roll_params:
            r, c = self.roll_params[dir]['r'], self.roll_params[dir]['c']
            distance = self.get_distance_from_location((r, c), dir)
            masks = dict()

            for d in range(self.distance_scope):
                masks[d] = np.where(distance == d)
            masks[self.distance_scope] = np.where(distance >= self.distance_scope)
            self.roll_params[dir]['distance_masks'] = masks

    def init_net(self):
        """
        Build and compile DNN.
        """
        self.model = Sequential([
            Dense(256, activation='relu', input_shape=(self.n_features,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1),
        ])
        adam = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(loss='mean_squared_error', optimizer=adam)


    def prep_ground_truth(self, next_features, rewards, batch_size=None):
        """
        Computes the target batch for training.
        :param next_features: Feature vectors of the current state and each possible action (batch_size, 3)
        :param rewards: Vector of rewards (batch_size,)
        :param batch_size: batch size
        :return:
        """
        if batch_size is None:
            batch_size = self.bs
        gt = np.zeros(batch_size)
        for i in range(batch_size):
            net_input = next_features[i]
            net_output = self.model.predict(net_input, batch_size=3)
            gt[i] = self.gamma*np.max(net_output) + rewards[i]
        return gt

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        the function for learning and improving the policy. it accepts the
        state-action-reward needed to learn from the final move of the game,
        and from that (and other state-action-rewards saved previously) it
        improves the policy.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
                          This is the final state of the round.
        :param too_slow: true if the game didn't get an action in time from the
                        policy for a few rounds in a row. you may use this to make your
                        computation time smaller (by lowering the batch size for example).
        """
        replace = self.replay_idx < self.bs
        choice = np.random.choice(range(self.replay_idx), size=self.bs, replace=replace)
        prev_batch = self.replay_prev[choice]
        next_batch = self.replay_next[choice]
        reward_batch = self.replay_reward[choice]
        action_batch = self.replay_action[choice]

        gt = self.prep_ground_truth(next_batch, reward_batch)

        self.model.train_on_batch(prev_batch, gt)

        if self.linear_takeover_on:
            self.linear_learn()

        try:
            if round % 100 == 0:
                if round > self.game_duration - self.score_scope:
                    self.log("Rewards in last 100 rounds which counts towards the score: " + str(
                        self.r_sum), 'VALUE')
                else:
                    self.log("Rewards in last 100 rounds: " + str(self.r_sum), 'VALUE')

                self.r_sum = 0
            else:
                self.r_sum += reward

        except Exception as e:
            self.log("Something Went Wrong...", 'EXCEPTION')
            self.log(e, 'EXCEPTION')

    def get_distance_from_location(self, location, direction):
        """
        Calculates the city block distance from a certain location, taking into account that the 
        board is cyclic and only L,R,F actions are allowed
        :param location: tuple of rows,cols - location on the board
        :param direction: one of 'N', 'S', 'E', 'W'
        :return: the distances
        """
        h, w = self.board_size
        cols_idx, rows_idx = np.meshgrid(range(w), range(h))
        down_idx, up_idx = rows_idx, np.flip(rows_idx, axis=0)
        right_idx, left_idx = cols_idx, np.flip(cols_idx, axis=1)

        loc_r, loc_c = location

        if direction == 'N':
            down_idx[1:, loc_c] += 2
        elif direction == 'S':
            up_idx[:-1, loc_c] += 2
        elif direction == 'E':
            left_idx[loc_r, :-1] += 2
        elif direction == 'W':
            right_idx[loc_r, 1:] += 2

        roll_down = np.roll(down_idx, loc_r, axis=0)
        roll_up = np.roll(up_idx, -(h - loc_r - 1), axis=0)
        vertical_distance = np.minimum(roll_down, roll_up)

        roll_right = np.roll(right_idx, loc_c, axis=1)
        roll_left = np.roll(left_idx, -(w - loc_c - 1), axis=1)
        horizontal_distance = np.minimum(roll_left, roll_right)

        distance = vertical_distance + horizontal_distance
        return distance


    def spatial_features(self, rolled_board, dir):
        """
        returns a one-hot encoding vector of a box around the center of the rolled board (which is
        the position of the snake's head after an action, as described in Answers.pdf).
        :param rolled_board: The game board, rolled
        :param dir: The snake's direction of movement (after the action)
        """
        r, c = self.roll_params[dir]['r'], self.roll_params[dir]['c']
        b = self.spatial_box
        box = rolled_board[r - b:r + b + 1, c - b:c + b + 1]
        box = np.rot90(box, k=self.rotate_num[dir]).flatten()
        assert box.size == ((b * 2) + 1) ** 2
        spatial_features = np.zeros(NUM_SYMBOLS * box.size)
        box_indices = np.array(range(box.size))*NUM_SYMBOLS
        spatial_features[box_indices+box+1] = 1
        return spatial_features

    def distance_features(self, state, action):
        """
        Computes the distance features - for each distance d in range(self.distance_scope), we count how
        many symbols of each type appear on the board with city block distance d from the 
        position of the snake after taking the specified action.

        Also computes the spatial features using the already-rolled board.

        Returns a vector of the distance & spatial features.
        """
        board, head = state
        head_pos, direction = head
        next_direction = bp.Policy.TURNS[direction][action]
        next_location = head_pos.move(next_direction)
        r = next_location[0]
        c = next_location[1]

        # roll board (np.take is 4x faster than actually rolling)
        rolled_board = board.take(self.radius_map[next_direction]['r'] + r, axis=0, mode='wrap').\
            take(self.radius_map[next_direction]['c'] + c, axis=1, mode='wrap')

        masks = self.roll_params[next_direction]['distance_masks']
        features = np.zeros((self.distance_scope + 1) * NUM_SYMBOLS)

        for d in range(self.distance_scope + 1):
            f = np.bincount(rolled_board[masks[d]] + 1, minlength=NUM_SYMBOLS)
            # Normalize. the feature is the percent of X out of the stuff which are in distance d (worked better)
            f = f / np.sum(f)
            features[d*NUM_SYMBOLS:(d+1)*NUM_SYMBOLS] = f

        # return features
        return np.concatenate((features, self.spatial_features(rolled_board, next_direction)))

    def get_features(self, state, action):
        """
        returns f(s,a).
        """
        return self.distance_features(state, action)

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        """
        the function for choosing an action, given current state.
        it accepts the state-action-reward needed to learn from the previous
        move (which it can save in a data structure for future learning), and
        accepts the new state from which it needs to decide how to act.
        :param round: the round of the game.
        :param prev_state: the previous state from which the policy acted.
        :param prev_action: the previous action the policy chose.
        :param reward: the reward given by the environment following the previous action.
        :param new_state: the new state that the agent is presented with, following the previous action.
        :param too_slow: true if the game didn't get an action in time from the
                        policy for a few rounds in a row. use this to make your
                        computation time smaller (by lowering the batch size for example)...
        :return: an action (from Policy.Actions) in response to the new_state.
        """
        # Manage epsilon-decay
        if round > self.linear_epsilon_decay_start:
            if self.linear_epsilon_decay_scope > 0:
                self.epsilon -= self.linear_epsilon_decay
                self.linear_epsilon_decay_scope -= 1

        if not self.first_act:
            # save (prev_state, prev_action, reward, new_state) to the linear policy's buffer:
            if self.linear_takeover_on:
                linear_action = self.linear_act(round, prev_state, prev_action, reward, new_state, too_slow)

            # update the index of the replay buffer
            idx = self.replay_idx
            if idx >= self.max_replay:
                idx = random.randint(0, self.max_replay - 1)
            else:
                self.replay_idx += 1

            # save (prev_state, prev_action, reward, new_state) to the replay buffer:
            self.replay_prev[idx] = self.get_features(prev_state, prev_action)
            next_features = np.zeros((3, self.n_features))
            for a in self.act2idx:
                next_features[self.act2idx[a]] = self.get_features(new_state, a)
            self.replay_next[idx] = next_features
            self.replay_action[idx] = self.act2idx[prev_action]
            self.replay_reward[idx] = reward

        if np.random.rand() <= self.epsilon or self.first_act:
            self.first_act = False
            return np.random.choice(bp.Policy.ACTIONS)

        elif self.linear_takeover_on:
            return linear_action

        # otherwise, choose action with DNN
        net_input = self.replay_next[idx]
        net_output = self.model.predict(net_input, batch_size=3)
        action_idx = np.argmax(net_output)
        action = self.idx2act[action_idx]

        return action
