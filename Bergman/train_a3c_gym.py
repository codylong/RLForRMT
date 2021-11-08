"""An example of training A3C against OpenAI Gym Envs.

This script is an example of training a A3C agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported.

To solve CartPole-v0, run:
    python train_a3c_gym.py 8 --env CartPole-v0

To solve InvertedPendulum-v1, run:
    python train_a3c_gym.py 8 --env InvertedPendulum-v1 --arch LSTMGaussian --t-max 50  # noqa
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()
import argparse
import os

# This prevents numpy from using multiple threads
os.environ['OMP_NUM_THREADS'] = '1'

# this causes the program to load the latest (upgraded) libraries from the local install rather than the old default packages in the global install on the cluster
# import sys
# sys.path.insert(1, "/users/ruehle/.local/lib/python2.7/site-packages/")

import chainer
from chainer import functions as F
from chainer import links as L
import gym
import gym_Bergman
#gym.undo_logger_setup()
import gym.wrappers
import numpy as np

from chainerrl.agents import a3c
from chainerrl import experiments
from chainerrl import links
from chainerrl import misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies
from chainerrl.recurrent import RecurrentChainMixin
from chainerrl import v_function


def phi(obs):
    return obs.astype(np.float32)


class A3CFFSoftmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward softmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
        self.pi = policies.SoftmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A3CFFMellowmax(chainer.ChainList, a3c.A3CModel):
    """An example of A3C feedforward mellowmax policy."""

    def __init__(self, ndim_obs, n_actions, hidden_sizes=(200, 200)):
        self.pi = policies.MellowmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


class A3CLSTMGaussian(chainer.ChainList, a3c.A3CModel, RecurrentChainMixin):
    """An example of A3C recurrent Gaussian policy."""

    def __init__(self, obs_size, action_size, hidden_size=200, lstm_size=128):
        self.pi_head = L.Linear(obs_size, hidden_size)
        self.v_head = L.Linear(obs_size, hidden_size)
        self.pi_lstm = L.LSTM(hidden_size, lstm_size)
        self.v_lstm = L.LSTM(hidden_size, lstm_size)
        self.pi = policies.LinearGaussianPolicyWithDiagonalCovariance(
            lstm_size, action_size)
        self.v = v_function.FCVFunction(lstm_size)
        super().__init__(self.pi_head, self.v_head,
                         self.pi_lstm, self.v_lstm, self.pi, self.v)

    def pi_and_v(self, state):

        def forward(head, lstm, tail):
            h = F.relu(head(state))
            h = lstm(h)
            return tail(h)

        pout = forward(self.pi_head, self.pi_lstm, self.pi)
        vout = forward(self.v_head, self.v_lstm, self.v)

        return pout, vout

class A3CLSTM(chainer.ChainList, a3c.A3CModel, RecurrentChainMixin):
    """FR: Just a simple modification of the LSTM layer that runs with discrete action spaces"""

    def __init__(self, obs_size, action_size, hidden_size=200, lstm_size=128):
        self.pi_head = L.Linear(obs_size, hidden_size)
        self.v_head = L.Linear(obs_size, hidden_size)
        self.pi_lstm = L.LSTM(hidden_size, lstm_size)
        self.v_lstm = L.LSTM(hidden_size, lstm_size)
        self.pi = policies.FCSoftmaxPolicy(lstm_size, action_size)
        self.v = v_function.FCVFunction(lstm_size)
        super().__init__(self.pi_head, self.v_head, self.pi_lstm, self.v_lstm, self.pi, self.v)

    def pi_and_v(self, state):
        def forward(head, lstm, tail):
            h = F.relu(head(state))
            h = lstm(h)
            return tail(h)

        pout = forward(self.pi_head, self.pi_lstm, self.pi)
        vout = forward(self.v_head, self.v_lstm, self.v)

        return pout, vout



def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('processes', type=int)
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--arch', type=str, default='FFSoftmax',
                        choices=('FFSoftmax', 'FFMellowmax', 'LSTMGaussian', 'LSTMFR', 'LSTMJH'))
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--t-max', type=int, default=5)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--steps', type=int, default=8 * 10 ** 7)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    parser.add_argument('--eval-n-runs', type=int, default=10)
    parser.add_argument('--rmsprop-epsilon', type=float, default=1e-1)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--logger-level', type=int, default=logging.ERROR)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--nmod', type=int, default=10)
    parser.add_argument('--moments', type=str, default='[0]')
    parser.add_argument('--numdraws', type=int, default=100)
    parser.add_argument('--stepsize', type=float, default=1e-2)
    parser.add_argument('--mean', type=float, default=0)
    parser.add_argument('--sigma', type=float, default=1e-2)
    parser.add_argument('--meantol', type=float, default=1e-6)
    parser.add_argument('--sigmatol', type=float, default=1e-6)
    parser.add_argument('--tol', type=float, default=1e-2)
    parser.add_argument('--weights', type=str, default='[]')
    args = parser.parse_args()

    logging.basicConfig(level=args.logger_level)
    mylog = logging.getLogger()
    mylog.propagate = False
    
    # Set a random seed used in ChainerRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    misc.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.processes) + args.seed * args.processes
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)

    def make_env(process_idx, test):
        env = gym.make(args.env)
        env.env.sigma = args.sigma
        env.env.weights = args.weights
        env.env.tol = args.tol
        env.env.nmod = args.nmod
        env.env.moments = args.moments
        env.env.numdraws = args.numdraws
        env.env.stepsize = args.stepsize
        env.env.mean = args.mean
        env.env.meantol = args.meantol
        env.env.sigmatol = args.sigmatol
        env.env.second_init()
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        if args.monitor and process_idx == 0:
            env = gym.wrappers.Monitor(env, args.outdir)
        # Scale rewards observed by agents
        # if not test:
        #     misc.env_modifiers.make_reward_filtered(
        #         env, lambda x: x * args.reward_scale_factor)
        # if args.render and process_idx == 0 and not test:
        #     misc.env_modifiers.make_rendered(env)
        return env

    sample_env = gym.make(args.env)
    sample_env.env.sigma = args.sigma
    sample_env.env.weights = args.weights
    sample_env.env.tol = args.tol
    sample_env.env.nmod = args.nmod
    sample_env.env.moments = args.moments
    sample_env.env.numdraws = args.numdraws
    sample_env.env.stepsize = args.stepsize
    sample_env.env.mean = args.mean
    sample_env.env.meantol = args.meantol
    sample_env.env.sigmatol = args.sigmatol
    sample_env.env.second_init()
    timestep_limit = sample_env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    sample_env.action_space = sample_env.env.action_space
    sample_env.observation_space = sample_env.env.observation_space
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space

    # Switch policy types accordingly to action space types
    if args.arch == 'LSTMGaussian':
        model = A3CLSTMGaussian(obs_space.low.size, action_space.low.size)
    elif args.arch == 'LSTMFR':
        model = A3CLSTM(obs_space.low.size, action_space.n)    
    elif args.arch == 'LSTMJH':
        model = A3CLSTM(obs_space.n, action_space.n)    
    elif args.arch == 'FFSoftmax':
        model = A3CFFSoftmax(obs_space.low.size, action_space.n)
        #model = A3CFFSoftmax(obs_space.n, action_space.n)
    elif args.arch == 'FFMellowmax':
        model = A3CFFMellowmax(obs_space.low.size, action_space.n)

    opt = rmsprop_async.RMSpropAsync(
        lr=args.lr, eps=args.rmsprop_epsilon, alpha=0.99)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(40))
    if args.weight_decay > 0:
        opt.add_hook(NonbiasWeightDecay(args.weight_decay))

    agent = a3c.A3C(model, opt, t_max=args.t_max, gamma=args.gamma,
                    beta=args.beta, phi=phi)
    if args.load:
        agent.load(args.load)

    if args.demo:
        env = make_env(0, True)
        env.env.setOutputFilePath(args.outdir)
        env.env.setProcessIdx(0)
        env.env.init_output()
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_runs=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_async_bergman(
            agent=agent,
            outdir=args.outdir,
            processes=args.processes,
            make_env=make_env,
            profile=args.profile,
            steps=args.steps,
            eval_n_runs=args.eval_n_runs,
            eval_interval=args.eval_interval,
            max_episode_len=timestep_limit)


if __name__ == '__main__':
    main()
