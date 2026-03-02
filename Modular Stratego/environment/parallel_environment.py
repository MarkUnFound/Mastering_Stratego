import torch.multiprocessing as mp
import numpy as np
import torch
from environment import StrategoEnvironment

def worker(remote, parent_remote, env_idx, device):
    try:
        parent_remote.close()
        # Initialize environment on CPU to avoid CUDA multiprocessing issues
        # We will move tensors to the specified device in the main process
        env = StrategoEnvironment(device='cpu')
        
        while True:
            try:
                cmd, data = remote.recv()
                
                if cmd == 'step':
                    action = data
                    next_state, reward, done, info = env.step(action)
                    # Get valid moves for the next state
                    valid_moves = env.get_valid_moves()
                    remote.send((next_state, reward, done, info, valid_moves))
                    
                elif cmd == 'reset':
                    placements = data
                    p1_placement = placements.get('p1_placement')
                    p2_placement = placements.get('p2_placement')
                    state = env.reset(p1_placement=p1_placement, p2_placement=p2_placement)
                    valid_moves = env.get_valid_moves()
                    # Return format matching step: (state, reward, done, info, valid_moves)
                    remote.send((state, 0.0, False, {}, valid_moves))
                    
                elif cmd == 'get_valid_moves':
                    moves = env.get_valid_moves()
                    remote.send(moves)
                    
                elif cmd == 'close':
                    remote.close()
                    break
                    
                elif cmd == 'get_attr':
                    attr_name = data
                    if hasattr(env, attr_name):
                        remote.send(getattr(env, attr_name))
                    else:
                        remote.send(None)
                        
                elif cmd == 'call_method':
                    method_name, args = data
                    if hasattr(env, method_name):
                        method = getattr(env, method_name)
                        result = method(*args)
                        remote.send(result)
                    else:
                        remote.send(None)
                        
                elif cmd == 'set_attr':
                    attr_name, value = data
                    setattr(env, attr_name, value)
                    remote.send(True)  # Acknowledge
                else:
                    print(f"Worker {env_idx}: Unknown command {cmd}")
                    remote.close()
                    break
            except EOFError:
                break
            except Exception as e:
                with open("worker_errors.txt", "a") as f:
                    f.write(f"Worker {env_idx} loop error: {e}\n")
                    import traceback
                    traceback.print_exc(file=f)
                print(f"Worker {env_idx} loop error: {e}")
                remote.close()
                break
    except Exception as e:
        with open("worker_errors.txt", "a") as f:
            f.write(f"Worker {env_idx} init error: {e}\n")
            import traceback
            traceback.print_exc(file=f)
        print(f"Worker {env_idx} init error: {e}")
        if 'remote' in locals():
            remote.close()

class ParallelStrategoEnvironment:
    def __init__(self, num_envs, device='cpu'):
        """
        Parallel environment wrapper for Stratego.
        
        Args:
            num_envs: Number of parallel environments
            device: Device to put the output tensors on ('cpu' or 'cuda')
        """
        self.num_envs = num_envs
        self.device = device
        ctx = mp.get_context('spawn')
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, i, device)) 
                   for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes))]
        
        for p in self.ps:
            p.daemon = True # terminate if main process dies
            p.start()
            
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions_or_commands):
        """
        Step the environments.
        
        Args:
            actions_or_commands: List of actions or reset commands.
                - If item is a tuple ('reset', placements), it resets that env.
                - Otherwise it's treated as an action for env.step().
                
        Returns:
            states: Tensor of states
            rewards: Tensor of rewards
            dones: Tensor of done flags
            infos: List of info dicts
            valid_moves: List of valid moves lists
        """
        for remote, cmd_data in zip(self.remotes, actions_or_commands):
            if isinstance(cmd_data, tuple) and len(cmd_data) == 2 and cmd_data[0] == 'reset':
                # It's a reset command: ('reset', {'p1_placement': ..., 'p2_placement': ...})
                remote.send(('reset', cmd_data[1]))
            else:
                # It's an action
                remote.send(('step', cmd_data))
                
        results = []
        for i, remote in enumerate(self.remotes):
            try:
                result = remote.recv()
                results.append(result)
            except (EOFError, BrokenPipeError) as e:
                print(f"\n  Worker {i} pipe broken in step: {e}")
                print(f"   This usually means a worker crashed. Restarting worker...")
                self._restart_worker(i)
                # Send reset to new worker and get result
                # We can't easily retry the step since we lost state, so we reset
                self.remotes[i].send(('reset', {}))
                result = self.remotes[i].recv()
                results.append(result)
        
        # Unzip results
        next_states, rewards, dones, infos, valid_moves = zip(*results)
        
        # Stack states and move to device
        # StrategoEnvironment returns GameState objects? 
        # Wait, env.step() returns next_game_state.
        # In train_dqn.py, agent.get_state_representation(game_state) is called.
        # So next_states here are GameState objects (or whatever env returns).
        # We cannot stack GameState objects into a tensor.
        # We return the list of GameState objects. The agent will handle batching.
        
        # Convert rewards and dones to tensors
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        return next_states, rewards_tensor, dones_tensor, infos, valid_moves

    def reset(self, p1_placements=None, p2_placements=None):
        """
        Reset all environments.
        
        Args:
            p1_placements: List of placements for P1 (optional)
            p2_placements: List of placements for P2 (optional)
        """
        for i, remote in enumerate(self.remotes):
            p1 = p1_placements[i] if p1_placements else None
            p2 = p2_placements[i] if p2_placements else None
            remote.send(('reset', {'p1_placement': p1, 'p2_placement': p2}))
            
        results = []
        for i, remote in enumerate(self.remotes):
            try:
                result = remote.recv()
                results.append(result)
            except (EOFError, BrokenPipeError) as e:
                print(f"\n  Worker {i} pipe broken in reset: {e}")
                print(f"   Restarting worker...")
                self._restart_worker(i)
                # Send reset to new worker
                p1 = p1_placements[i] if p1_placements else None
                p2 = p2_placements[i] if p2_placements else None
                self.remotes[i].send(('reset', {'p1_placement': p1, 'p2_placement': p2}))
                result = self.remotes[i].recv()
                results.append(result)

        states, rewards, dones, infos, valid_moves = zip(*results)
        
        return states, rewards, dones, infos, valid_moves

    def _restart_worker(self, i):
        """Helper to restart a crashed worker"""
        # Close the broken process
        if self.ps[i].is_alive():
            self.ps[i].terminate()
        self.ps[i].join(timeout=1)
        
        # Create a new worker
        ctx = mp.get_context('spawn')
        parent_remote, work_remote = ctx.Pipe()
        
        # Update remotes list
        remotes_list = list(self.remotes)
        remotes_list[i] = parent_remote
        self.remotes = tuple(remotes_list)
        
        p = ctx.Process(target=worker, args=(work_remote, parent_remote, i, self.device))
        p.daemon = True
        p.start()
        self.ps[i] = p
        work_remote.close()

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            
    def get_attr(self, attr_name):
        # Get attribute from first env (assuming homogeneous)
        self.remotes[0].send(('get_attr', attr_name))
        return self.remotes[0].recv()

    def call_method(self, method_name, *args):
        """Call a method on the first environment and return the result."""
        self.remotes[0].send(('call_method', (method_name, args)))
        return self.remotes[0].recv()
    
    def set_full_observability(self, enabled: bool):
        """
        Set full observability mode for all environments.
        Used for Phase 1 curriculum (full observability training).
        """
        for remote in self.remotes:
            remote.send(('call_method', ('set_full_observability', (enabled,))))
        # Receive acknowledgments
        for remote in self.remotes:
            remote.recv()
    
    def set_max_turns(self, max_turns: int):
        """
        Set max turns limit for all environments.
        Used for curriculum-based game length limits.
        """
        for remote in self.remotes:
            remote.send(('set_attr', ('max_turns', max_turns)))
        # Receive acknowledgments
        for remote in self.remotes:
            remote.recv()
