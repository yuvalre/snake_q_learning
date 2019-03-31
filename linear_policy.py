from policies import base_policy as bp
import numpy as np

LEARNING_TIME = bp.LEARNING_TIME

EPSILON = 0.5
NUM_SYMBOLS = 11
GAMMA = 0.5
LEARNING_RATE = 0.01
STEPS_AHEAD = 2
NUM_ACTIONS = 3


class Linear308490234(bp.Policy):
    """
    A linear Q-Learning approximation which avoids collisions with obstacles and other snakes and
    wishes to eat good yummy fruits. It has an epsilon parameter which controls the
    percentage of actions which are randomly chosen.
    """

    def cast_string_args(self, policy_args):
        policy_args['epsilon'] = float(
            policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        policy_args['gamma'] = float(
            policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        policy_args['lr'] = float(
            policy_args['lr']) if 'lr' in policy_args else LEARNING_RATE
        return policy_args

    def init_run(self):
        self.r_sum = 0
        self.first_act = True # Flag for first round

        self.rotate_num = {'N': 0, 'E': 1, 'S': 2, 'W': 3}  # How many 90deg rotates to perform until we're at N

        self.n_features = NUM_SYMBOLS + ((STEPS_AHEAD - 1) * NUM_ACTIONS * NUM_SYMBOLS) + 1

        # buffer
        #########
        self.buffi = 0
        self.feature_buff = np.zeros(shape=(LEARNING_TIME * 2, self.n_features))
        self.deltas = np.zeros(shape=(LEARNING_TIME * 2, 1))
        self.w = np.zeros(self.n_features)

    def get_q(self, state, action):
        """
        Returns the feature vector f(s,a) and current estimation of Q(s,a).
        The feature vector is a one-hot encoding vector of the values of the next 2 steps:
        - the value at the next position (where we'll be after the specified action)
        - and the values of the 3 possible next-next-turn positions after that action.
        """
        features = np.zeros(self.n_features)

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
        indices.append(self.n_features-1)

        return features, self.w[indices].sum()

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
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
        self.deltas = self.deltas[:self.buffi]
        self.feature_buff = self.feature_buff[:self.buffi]
        self.buffi = 0
        self.w = self.w - self.lr * (self.deltas * self.feature_buff).mean(axis=0)

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
        board, head = new_state
        head_pos, direction = head

        if round > self.game_duration - self.score_scope:
            self.epsilon = 0.0

        if not self.first_act:
            prev_features, prev_q = self.get_q(prev_state, prev_action)
            self.feature_buff[self.buffi] = prev_features

            # calculate delta and best act:
            qs = []
            acts = list(np.random.permutation(bp.Policy.ACTIONS))
            for a in acts:
                features, q = self.get_q(new_state, a)
                qs.append(q)
            best_act = acts[np.argmax(qs)]
            best_q = np.max(qs)

            delta = prev_q - (reward + (self.gamma * best_q))
            self.deltas[self.buffi] = delta
            self.buffi += 1

        if np.random.rand() < self.epsilon or self.first_act:
            self.first_act = False
            return np.random.choice(bp.Policy.ACTIONS)

        return best_act
