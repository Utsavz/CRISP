import numpy as np

def make_sample_her_transitions(replay_strategy, replay_k, reward_fun, hrl_scope, hac, dac):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if hac and hrl_scope=="ddpg0":
        future_p = 0
    elif replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0
    # if hrl_scope != 'ddpg0':
    #     future_p = 0
    env_name = "kitchen"

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """

        rollout_batch_size = len(episode_batch['u'])
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size) # selecting from among the rollout_batch_sizes(size of the buffer)

        epi_len = np.array([episode_batch['u'][x].shape[0] for x in episode_idxs])
        t_samples = np.random.uniform(0,epi_len-1,batch_size).astype(int)
        t_sam = t_samples
        
        
        transitions = {}        
        for key in episode_batch.keys():
            if key == 'info_is_success':
                continue

            transitions[key] = np.array([episode_batch[key][episode][sample] for episode,sample in zip(episode_idxs,t_samples)])
        

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)[0] # Select future_p percent of all the indices(size = batch_size)
        future_offset = np.random.uniform(size=batch_size) * (epi_len - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        for i, idx in enumerate(episode_idxs[her_indexes]):
            if len(episode_batch['ag'][idx]) > future_t[i]:
                transitions['g'][her_indexes[i]] = episode_batch['ag'][idx][future_t[i]]

        # Reconstruct info dictionary for reward  computation.
        info = {}
        for key, value in transitions.items():
            if key.startswith('info_'):
                info[key.replace('info_', '')] = value

        # Re-compute reward since we may have substituted the goal.
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        reward_params['info'] = info
        if hac:
            transitions['r'] = reward_fun(**reward_params) + transitions['r'].reshape(-1)
        elif future_p == 0:
            transitions['r'] = transitions['r'].reshape(-1)
        else:
            transitions['r'] = reward_fun(**reward_params)# + transitions['r'].reshape(-1)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['u'].shape[0] == batch_size_in_transitions)
        return transitions

    return _sample_her_transitions
