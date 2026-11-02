"""Step 5: the learner (C51 target, masked Double DQN, one gamma, target copies, PER, NoisyNet acting)."""

import numpy as np
import pytest
import torch

from action_encoding import NUM_ACTIONS, encode_move, legal_action_mask, mirror_action
from environment import StrategoEnvironment


def _agent(cpu, **kwargs):
    from drqn_agent import RainbowAgent
    kwargs.setdefault('num_envs', 1)
    kwargs.setdefault('buffer_size', 200)
    return RainbowAgent(player_id=1, device=cpu, **kwargs)


def _same_weights(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return all(torch.equal(sa[k], sb[k]) for k in sa)


# ---------------------------------------------------------------- C51 projection

def test_projection_keeps_all_the_mass_for_terminal_and_bootstrapped_targets(cpu, tiny_network):
    """A terminal draw (r = 0 lands exactly on the middle atom) got an all-zero target."""
    agent = _agent(cpu)
    rewards = torch.tensor([1.0, -1.0, 0.0, 0.0, 0.4, -10.0, 10.0, 0.123])
    dones = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    probs = torch.softmax(torch.randn(len(rewards), agent.num_atoms), dim=1)
    m = agent._project_distribution(probs, rewards, dones, agent.gamma_n)
    assert torch.allclose(m.sum(dim=1), torch.ones(len(rewards)), atol=1e-5)
    middle = agent.num_atoms // 2
    assert m[2, middle].item() == pytest.approx(1.0)                     # draw: all mass on z = 0
    assert m[5, 0].item() == pytest.approx(1.0) and m[6, -1].item() == pytest.approx(1.0)  # support ends


def test_projection_of_a_terminal_reward_has_the_reward_as_its_mean(cpu, tiny_network):
    agent = _agent(cpu)
    probs = torch.full((3, agent.num_atoms), 1.0 / agent.num_atoms)
    rewards = torch.tensor([0.3, -0.7, 0.0])
    m = agent._project_distribution(probs, rewards, torch.ones(3), agent.gamma_n)
    assert torch.allclose((m * agent.support).sum(dim=1), rewards, atol=1e-5)


# ---------------------------------------------------------------- masked Double DQN

def test_double_dqn_picks_the_best_legal_next_action(cpu, tiny_network):
    agent = _agent(cpu)
    torch.manual_seed(0)
    log_probs = torch.log_softmax(torch.randn(4, NUM_ACTIONS, agent.num_atoms), dim=2)
    legal = torch.rand(4, NUM_ACTIONS) < 0.01
    chosen = agent._double_dqn_next_actions(log_probs, legal)
    q = (log_probs.exp() * agent.support).sum(dim=2)
    for i in range(4):
        assert legal[i, chosen[i]]
        assert q[i, chosen[i]] == q[i][legal[i]].max()
    unmasked = agent._double_dqn_next_actions(log_probs, None)
    assert torch.equal(unmasked, q.argmax(dim=1))


def _steps(cpu, n):
    """n consecutive (state, move, next_state, next legal moves) of a random game, Player 1's view."""
    env = StrategoEnvironment(cpu)
    state = env.reset()
    rng = np.random.default_rng(0)
    out = []
    while len(out) < n:
        moves = env.get_valid_moves()
        move = moves[rng.integers(len(moves))]
        next_state, _, done, _ = env.step(move)
        assert not done
        reply = env.get_valid_moves()
        after_reply, _, done, _ = env.step(reply[rng.integers(len(reply))])
        assert not done
        out.append((state, move, after_reply, env.get_valid_moves()))
        state = after_reply
    return out


def test_next_state_legal_moves_reach_the_replay_buffer_through_n_steps_and_the_flip(cpu, tiny_network, config_override):
    config_override(ENABLE_DATA_AUGMENTATION=True, AUGMENTATION_TYPES=["flip"])
    agent = _agent(cpu)
    n = agent.n_steps
    steps = _steps(cpu, n)
    for state, move, next_state, next_moves in steps:
        agent.remember_batch([state.board], [move], [0.0], [next_state.board], [False], [True], [state], [next_state],
                             infos=[{}], next_legal_moves=[next_moves])
    stored = [agent.memory.tree.data[i] for i in range(len(agent.memory))]
    assert len(stored) == 2  # one n-step transition and its mirror image
    original, flipped = stored
    expected = legal_action_mask(steps[-1][3], 1)  # the legal moves after the n-th step
    assert np.array_equal(np.unpackbits(original.next_legal, count=NUM_ACTIONS).astype(bool), expected)
    mirrored = np.zeros(NUM_ACTIONS, dtype=bool)
    mirrored[[mirror_action(a) for a in np.flatnonzero(expected)]] = True
    assert np.array_equal(np.unpackbits(flipped.next_legal, count=NUM_ACTIONS).astype(bool), mirrored)
    assert flipped.action == mirror_action(original.action) and original.action == encode_move(steps[0][1], 1)

    sample = agent.memory.sample(4)
    next_legal = sample[-1]
    assert next_legal.dtype == torch.bool and next_legal.shape == (len(sample[0]), NUM_ACTIONS)


def test_terminal_transitions_store_no_mask_and_allow_every_action(cpu, tiny_network):
    agent = _agent(cpu)
    state, move, next_state, next_moves = _steps(cpu, 1)[0]
    agent.remember_batch([state.board], [move], [1.0], [next_state.board], [True], [True], [state], [next_state],
                         infos=[{}], next_legal_moves=[next_moves])
    assert all(agent.memory.tree.data[i].next_legal is None for i in range(len(agent.memory)))
    assert agent.memory.sample(2)[-1] is None


def test_replay_masks_the_double_dqn_action(cpu, tiny_network, config_override, monkeypatch):
    config_override(WARMUP_STEPS=0)
    agent = _agent(cpu, batch_size=4)
    for state, move, next_state, next_moves in _steps(cpu, agent.n_steps + 4):
        agent.remember_batch([state.board], [move], [0.0], [next_state.board], [False], [True], [state], [next_state],
                             infos=[{}], next_legal_moves=[next_moves])
    seen = []
    original = agent._double_dqn_next_actions

    def spy(log_probs, next_legal):
        actions = original(log_probs, next_legal)
        seen.append((next_legal, actions))
        return actions

    monkeypatch.setattr(agent, '_double_dqn_next_actions', spy)
    assert agent.replay() is not None
    [(next_legal, actions)] = seen
    assert next_legal is not None
    assert next_legal[torch.arange(len(actions)), actions].all()


# ---------------------------------------------------------------- one gamma

def test_one_discount_factor(cpu, tiny_network):
    from training_config import GAMMA, N_STEPS
    agent = _agent(cpu)
    assert agent.gamma == GAMMA
    assert agent.n_step_buffers[0].gamma == GAMMA
    assert agent.gamma_n == pytest.approx(GAMMA ** N_STEPS)


# ---------------------------------------------------------------- target network

def _fill(agent, n=40):
    c = agent.input_channels
    for i in range(n):
        agent.memory.add(torch.rand(c, 10, 10), i % agent.action_size, 0.1, torch.rand(c, 10, 10), i % 7 == 0)


def test_target_starts_as_a_copy_and_is_hard_copied_every_n_updates(cpu, tiny_network, config_override):
    config_override(WARMUP_STEPS=0, TARGET_UPDATE_INTERVAL=3)
    agent = _agent(cpu, batch_size=8)
    assert _same_weights(agent.q_network, agent.target_network)
    _fill(agent)
    for _ in range(2):
        assert agent.replay() is not None
    assert not _same_weights(agent.q_network, agent.target_network)
    assert agent.replay() is not None
    assert agent.gradient_updates == 3
    assert _same_weights(agent.q_network, agent.target_network)


# ---------------------------------------------------------------- PER

def test_new_samples_get_the_largest_priority_so_far():
    from prioritized_memory import PrioritizedReplayBuffer
    buf = PrioritizedReplayBuffer(100, device='cpu', alpha=0.6)
    x = torch.zeros(30, 10, 10)
    buf.add(x, 0, 0.0, x, False)
    first = buf.capacity - 1
    buf.update_priorities([first], [4.0])
    buf.add(x, 1, 0.0, x, False)
    assert buf.tree.tree[first] == pytest.approx(4.0 ** 0.6)
    assert buf.tree.tree[first + 1] == pytest.approx(4.0 ** 0.6)  # the thesis code gave 4 ** 0.36


# ---------------------------------------------------------------- NoisyNet acting

def test_acting_uses_fresh_noise_and_greedy_acting_uses_the_mean_weights(cpu, tiny_network):
    agent = _agent(cpu)
    state = StrategoEnvironment(cpu).reset()
    x = agent.get_batch_state_representation([state.board], [state])
    a, b = agent._q_forward_for_acting(x), agent._q_forward_for_acting(x)
    assert not torch.equal(a, b)
    assert agent.q_network.training and not agent.q_network.advantage_out.noisy

    agent.noisy_acting = False
    c, d = agent._q_forward_for_acting(x), agent._q_forward_for_acting(x)
    assert torch.equal(c, d)
    agent.q_network.eval()
    with torch.no_grad():
        assert torch.allclose(agent.q_network(x), c)


def test_act_batch_resamples_the_noise(cpu, tiny_network, config_override):
    config_override(IMITATION_ENABLED=False)
    agent = _agent(cpu)
    env = StrategoEnvironment(cpu)
    state = env.reset()
    before = agent.q_network.advantage_out.weight_epsilon.clone()
    assert agent.act_batch([state.board], [env.get_valid_moves()], [state], env_indices=[0])[0] in env.get_valid_moves()
    assert not torch.equal(before, agent.q_network.advantage_out.weight_epsilon)


def test_replay_resamples_the_noise(cpu, tiny_network, config_override):
    config_override(WARMUP_STEPS=0)
    agent = _agent(cpu, batch_size=8)
    _fill(agent)
    before = agent.target_network.advantage_out.weight_epsilon.clone()
    agent.replay()
    assert not torch.equal(before, agent.target_network.advantage_out.weight_epsilon)


# ---------------------------------------------------------------- defaults

def test_plain_rainbow_loss_and_exact_legal_argmax_by_default(cpu, tiny_network):
    agent = _agent(cpu)
    assert not agent.kl_reg_enabled and not agent.entropy_reg_enabled and not agent.dynamic_damping
    assert not agent.use_heuristic_filter


def test_rotate_augmentation_is_rejected(cpu, tiny_network, config_override):
    """Rotating the canonical frame 180 degrees puts your pieces on the opponent's side."""
    config_override(ENABLE_DATA_AUGMENTATION=True, AUGMENTATION_TYPES=["flip", "rotate"])
    with pytest.raises(ValueError, match="rotate"):
        _agent(cpu)


# ---------------------------------------------------------------- uint8 storage

def test_observations_are_stored_as_uint8_and_read_back(cpu, tiny_network):
    from prioritized_memory import pack_state, unpack_states
    agent = _agent(cpu)
    state = StrategoEnvironment(cpu).reset()
    x = agent.get_batch_state_representation([state.board], [state])[0]
    x[27:] = torch.rand(3, 10, 10)  # the fraction planes
    packed = pack_state(x, 'cpu', torch.float32)
    assert packed.dtype == torch.uint8
    back = unpack_states([packed], 'cpu')[0]
    assert torch.equal(back[:27], x[:27])
    assert (back - x).abs().max() <= 1 / 510 + 1e-6
    y = torch.randn(30, 10, 10)
    assert torch.equal(unpack_states([pack_state(y, 'cpu', torch.float32)], 'cpu')[0], y)


# ---------------------------------------------------------------- turn tracker

def test_turn_tracker_hands_over_the_movers_next_legal_moves():
    from network import TurnTracker
    tt = TurnTracker(1)
    assert tt.record(0, 1, 's1', 'a', 's2', False, {}, next_moves=['p2 moves']) == []
    [t] = tt.record(0, -1, 's2', 'b', 's3', False, {}, next_moves=['p1 moves'])
    assert t['player'] == 1 and t['next_moves'] == ['p1 moves']
    ts = tt.record(0, 1, 's3', 'c', 's4', True, {'winner': 1}, next_moves=[])
    assert [t['next_moves'] for t in ts] == [None, None]


def test_trainer_stores_next_legal_moves_of_the_learner(tmp_path, config_override, monkeypatch):
    from network import RainbowAgent
    from test_trainer_smoke import run_tiny_training
    calls = []
    original = RainbowAgent.remember_batch

    def spy(self, *args, **kwargs):
        if self.player_id == 1:
            calls.append((args[3], args[4], kwargs.get('next_legal_moves')))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RainbowAgent, 'remember_batch', spy)
    run_tiny_training(tmp_path, config_override, monkeypatch, episodes=2)
    checked = 0
    for next_boards, dones, next_moves in calls:
        assert next_moves is not None
        for board, done, moves in zip(next_boards, dones, next_moves):
            if done:
                assert moves is None
                continue
            assert moves
            for (r, c), _ in moves:
                assert board[r][c] > 0  # Player 1's own piece
            checked += 1
    assert checked > 0
