from gym.envs.registration import register


register(
    id='bergman-v0',
    entry_point='gym_Bergman.bergman:Bergman',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 20000},
)

register(
    id='bergmanvmeans-v0',
    entry_point='gym_Bergman.bergmanvmeans:Bergman',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 20000},
)

register(
    id='wishart-v0',
    entry_point='gym_Wishart.wishart:Wishart',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 20000},
)

register(
    id='wishart-10k-v0',
    entry_point='gym_Wishart.wishart:Wishart',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
)

register(
    id='wishart-50k-v0',
    entry_point='gym_Wishart.wishart:Wishart',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 50000},
)

register(
    id='wishart-100k-v0',
    entry_point='gym_Wishart.wishart:Wishart',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 100000},
)

register(
    id='cc-inout2-v0',
    entry_point='gym_cc.cc_inout2:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
)

register(
    id='cc-inout-v0',
    entry_point='gym_cc.cc_inout:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
)

register(
    id='cc-MCTS-v0',
    entry_point='gym_cc.cc_MCTS:CCMCTS',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
)

register(
    id='cc-MCTS-10k-v0',
    entry_point='gym_cc.cc_MCTS:CCMCTS',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
)

register(
    id='cc-MCTS-20k-v0',
    entry_point='gym_cc.cc_MCTS:CCMCTS',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 20000},
)

register(
    id='cc-MCTS-50k-v0',
    entry_point='gym_cc.cc_MCTS:CCMCTS',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 50000},
)

register(
    id='cc-MCTS-100k-v0',
    entry_point='gym_cc.cc_MCTS:CCMCTS',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 100000},
)

register(
    id='cc-non-origin-start-1m-v0',
    entry_point='gym_cc.cc_eig_non_origin_start:CCNonOriginStart',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000000},
)

register(
    id='cc-eig-and-bump-MCTS-v0',
    entry_point='gym_cc.cc_eig_and_bump:CCEigAndBumpMCTS',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 20000},
)

register(
    id='cc-eig-and-bump-v0',
    entry_point='gym_cc.cc_eig_and_bump:CCEigAndBump',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 20000},
)

register(
    id='cc-eig-and-bump-10k-v0',
    entry_point='gym_cc.cc_eig_and_bump:CCEigAndBump',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
)
register(
    id='cc-eig-and-bump-20k-v0',
    entry_point='gym_cc.cc_eig_and_bump:CCEigAndBump',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 20000},
)
register(
    id='cc-eig-and-bump-30k-v0',
    entry_point='gym_cc.cc_eig_and_bump:CCEigAndBump',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 30000},
)
register(
    id='cc-eig-and-bump-40k-v0',
    entry_point='gym_cc.cc_eig_and_bump:CCEigAndBump',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 40000},
)
register(
    id='cc-eig-and-bump-50k-v0',
    entry_point='gym_cc.cc_eig_and_bump:CCEigAndBump',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 50000},
)
register(
    id='cc-eig-and-bump-60k-v0',
    entry_point='gym_cc.cc_eig_and_bump:CCEigAndBump',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 60000},
)
register(
    id='cc-eig-and-bump-70k-v0',
    entry_point='gym_cc.cc_eig_and_bump:CCEigAndBump',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 70000},
)
register(
    id='cc-eig-and-bump-80k-v0',
    entry_point='gym_cc.cc_eig_and_bump:CCEigAndBump',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 80000},
)
register(
    id='cc-eig-and-bump-90k-v0',
    entry_point='gym_cc.cc_eig_and_bump:CCEigAndBump',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 90000},
)
register(
    id='cc-eig-and-bump-100k-v0',
    entry_point='gym_cc.cc_eig_and_bump:CCEigAndBump',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 100000},
)
register(
    id='cc-eig-and-bump-200k-v0',
    entry_point='gym_cc.cc_eig_and_bump:CCEigAndBump',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 200000},
)
register(
    id='cc-eig-and-bump-500k-v0',
    entry_point='gym_cc.cc_eig_and_bump:CCEigAndBump',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 500000},
)
register(
    id='cc-eig-and-bump-1m-v0',
    entry_point='gym_cc.cc_eig_and_bump:CCEigAndBump',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000000},
)

###

register(
    id='cc-v0',
    entry_point='gym_cc.cc:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 20000},
)

register(
    id='cc-10k-v0',
    entry_point='gym_cc.cc:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
)

register(
    id='cc-20k-v0',
    entry_point='gym_cc.cc:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 20000},
)

register(
    id='cc-30k-v0',
    entry_point='gym_cc.cc:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 30000},
)

register(
    id='cc-40k-v0',
    entry_point='gym_cc.cc:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 40000},
)

register(
    id='cc-50k-v0',
    entry_point='gym_cc.cc:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 50000},
)

register(
    id='cc-60k-v0',
    entry_point='gym_cc.cc:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 60000},
)

register(
    id='cc-70k-v0',
    entry_point='gym_cc.cc:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 70000},
)

register(
    id='cc-80k-v0',
    entry_point='gym_cc.cc:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 80000},
)

register(
    id='cc-90k-v0',
    entry_point='gym_cc.cc:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 90000},
)

register(
    id='cc-100k-v0',
    entry_point='gym_cc.cc:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 100000},
)

register(
    id='cc-quickrun-v0',
    entry_point='gym_cc.cc:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 200},
)

register(
    id='cc-200k-v0',
    entry_point='gym_cc.cc:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 200000},
)

register(
    id='cc-500k-v0',
    entry_point='gym_cc.cc:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 500000},
)

register(
    id='cc-1m-v0',
    entry_point='gym_cc.cc:CC',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000000},
)

