import numpy as np

from gym import utils
from gym.envs.toy_text import discrete
from utils import poisson

# import sys
# from six import StringIO, b


class JacksCarRental(discrete.DiscreteEnv):
    def __init__(self,
                 capacity=20,
                 init_cars=12,
                 lam_a=[3, 3],
                 lam_b=[4, 2],
                 move_cost=-2,
                 max_move=5,
                 rent_income=10):

        self.capacity = capacity
        self.capacity_plus = capacity + 1
        self.init_cars = init_cars
        self.lam_a = lam_a
        self.lam_b = lam_b

        self.move_cost = move_cost
        self.move_max = max_move
        self.rent_income = rent_income

        self.nS = (capacity + 1) ^ 2
        self.nA = 2 * max_move + 1

        self._possible_outcomes = np.full((capacity + 1, capacity + 1), [])

        self._init_p()

    def _to_ab(self, s):
        return (s / self.capacity_plus, s % self.capacity_plus)

    def _to_s(self, a, b):
        return a * self.capacity_plus + b

    def _init_p(self):
        P = np.full((self.nS, self.nA), [])
        for act in range(-self.move_max, self.move_max + 1):
            for cur_a in range(self.capacity):
                for cur_b in range(self.capacity):
                    cur_s = self._to_s(cur_a, cur_b)
                    real_act = act
                    after_a = min(cur_a - act, self.capacity)
                    after_b = min(cur_b + act, self.capacity)
                    if after_a < 0:
                        real_act = cur_a

                    if after_b < 0:
                        real_act = -cur_b

                    # The reason we don't guard against overflow (after_a/b > capacity)
                    # is that with act larger, it incurs more costs and less available cars
                    # An optimal policy will try to minimize the costs and maximize cars

                    after_a = min(cur_a - real_act, self.capacity)
                    after_b = min(cur_b + real_act, self.capacity)

                    P[cur_s][act] = self._possible_rent_return_outcomes(
                        after_a, after_b, real_act)
        return P

    def _possible_rent_return_outcomes(self, a, b, act):
        all_rent_rewards = []
        if self._possible_outcomes[a][b]:
            all_rent_rewards = self._possible_outcomes[a][b]
        else:
            pA, pB = (poisson.poisson_possibility_split(self.lam_a[0], a),
                      poisson.poisson_possibility_split(self.lam_b[0], b))
            for ca, pa in enumerate(pA):
                for cb, pb in enumerate(pB):
                    cur_a = a - ca
                    cur_b = b - cb
                    prA, prB = (
                        poisson.poisson_possibility_split(
                            self.lam_a[1], self.capacity - cur_a),
                        poisson.poisson_possibility_split(
                            self.lam_b[1], self.capacity - cur_b),
                    )
                    for ra, pra in enumerate(prA):
                        for rb, prb in enumerate(prB):
                            cur_a += ra
                            cur_b += rb
                            next_s = self._to_s(cur_a, cur_b)
                            all_rent_rewards.append((
                                pa * pb * pra * prb,
                                next_s,
                                (pa + pb) * self.rent_income,
                                False,
                            ))
            self._possible_outcomes[a][b] = all_rent_rewards

        base_rew = act * self.move_cost
        return [(p, n, r + base_rew, t) for p, n, r, t in all_rent_rewards]
