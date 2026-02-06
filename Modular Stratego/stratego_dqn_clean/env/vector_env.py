
import torch.multiprocessing as mp
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
import traceback
from .environment import StrategoEnvironment

def worker(remote, parent_remote, env_idx, device_str, max_turns, strict_validation, safe_guards):
    """
    Worker process for running a single environment.
    """
    try:
        parent_remote.close()
        # Initialize environment on CPU to avoid CUDA multiprocessing issues
        # We will move tensors to the specified device in the main process
        env = StrategoEnvironment(
            device='cpu', 
            max_turns=max_turns,
            strict_validation=strict_validation,
            safe_guards=safe_guards
        )
        
        while True:
            try:
                cmd, data = remote.recv()
                
                if cmd == 'step':
                    action = data
                    next_state, reward, done, info = env.step(action)
                    
                    if done:
                        # Auto-reset logic
                        # Save the terminal state in info for learning
                        # Extract observable board from terminal state (which is a GameState object)
                        # The environment might have already reset internally? No, gym semantics.
                        
                        # Fix: Get board from env BEFORE reset
                        term_board = env.board.get_visible_board(1)
                        info['terminal_observation'] = term_board
                        
                        # Reset immediately to fresh state
                        env.reset()
                        next_board = env.board.get_visible_board(1)
                    else:
                        next_board = env.board.get_visible_board(1)
                        
                    # Get valid moves for the NEXT state
                    valid_moves = env.get_valid_moves()
                    
                    remote.send((next_board, reward, done, info, valid_moves))
                    
                elif cmd == 'reset':
                    placements = data
                    p1_placement = placements.get('p1_placement')
                    p2_placement = placements.get('p2_placement')
                    
                    env.reset(p1_placement=p1_placement, p2_placement=p2_placement)
                    state = env.board.get_visible_board(1)
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
                print(f"Worker {env_idx} loop error: {e}")
                traceback.print_exc()
                remote.close()
                break
                
    except Exception as e:
        print(f"Worker {env_idx} init error: {e}")
        traceback.print_exc()
        if 'remote' in locals():
            remote.close()

class VectorStrategoEnv:
    """
    Vectorized environment wrapper for Stratego.
    Handles parallel execution of multiple StrategoEnvironment instances.
    Implements AUTO-RESET: when an environment is done, it resets immediately.
    """
    def __init__(self, num_envs: int, device: str = 'cpu', max_turns: int = 1000, 
                 strict_validation: bool = False, safe_guards: bool = True):
        """
        Args:
            num_envs: Number of parallel environments
            device: Device for tensors
            max_turns: Max turns per episode
            strict_validation: Enable strict move validation
            safe_guards: Enable fast move safeguards
        """
        self.num_envs = num_envs
        self.device = device
        self.max_turns = max_turns
        self.strict_validation = strict_validation
        self.safe_guards = safe_guards
        
        ctx = mp.get_context('spawn')
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, i, device, max_turns, strict_validation, safe_guards)) 
                   for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes))]
        
        for p in self.ps:
            p.daemon = True # terminate if main process dies
            p.start()
            
        for remote in self.work_remotes:
            remote.close()
            
        print(f"[VectorStrategoEnv] Initialized {num_envs} environments")

    def step(self, actions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict], List[List]]:
        """
        Step all environments.
        
        Args:
            actions: List of actions, one per environment
            
        Returns:
            next_states: (N, 15, 10, 10) tensor - Reset states for done envs
            rewards: (N,) tensor
            dones: (N,) tensor
            infos: List of info dicts (containing 'terminal_observation' if done)
            valid_moves: List of valid moves lists for next_states
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
                
        results = []
        for i, remote in enumerate(self.remotes):
            try:
                result = remote.recv()
                results.append(result)
            except (EOFError, BrokenPipeError) as e:
                print(f"\n⚠️  Worker {i} pipe broken in step: {e}")
                print(f"   Restarting worker...")
                self._restart_worker(i)
                self.remotes[i].send(('reset', {}))
                result = self.remotes[i].recv()
                results.append(result)
        
        # Unzip results
        next_states_list, rewards, dones, infos, valid_moves = zip(*results)
        
        # list of numpy/torch -> stacked tensor
        # env.step returns numpy arrays, we convert to tensor here
        next_states_tensor = torch.stack([
            torch.as_tensor(s, dtype=torch.float32) for s in next_states_list
        ]).to(self.device)
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        return next_states_tensor, rewards_tensor, dones_tensor, infos, valid_moves

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
                print(f"\n⚠️  Worker {i} pipe broken in reset: {e}")
                print(f"   Restarting worker...")
                self._restart_worker(i)
                p1 = p1_placements[i] if p1_placements else None
                p2 = p2_placements[i] if p2_placements else None
                self.remotes[i].send(('reset', {'p1_placement': p1, 'p2_placement': p2}))
                result = self.remotes[i].recv()
                results.append(result)

        states_list, rewards, dones, infos, valid_moves = zip(*results)
        
        states_tensor = torch.stack([
            torch.as_tensor(s, dtype=torch.float32) for s in states_list
        ]).to(self.device)
        
        # rewards/dones are dummies from reset, likely unused
        
        return states_tensor, valid_moves

    def _restart_worker(self, i):
        """Helper to restart a crashed worker"""
        if self.ps[i].is_alive():
            self.ps[i].terminate()
        self.ps[i].join(timeout=1)
        
        ctx = mp.get_context('spawn')
        parent_remote, work_remote = ctx.Pipe()
        
        remotes_list = list(self.remotes)
        remotes_list[i] = parent_remote
        self.remotes = tuple(remotes_list)
        
        p = ctx.Process(target=worker, args=(work_remote, parent_remote, i, 'cpu', self.max_turns, self.strict_validation, self.safe_guards))
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
        self.remotes[0].send(('get_attr', attr_name))
        return self.remotes[0].recv()

    def call_method(self, method_name, *args):
        self.remotes[0].send(('call_method', (method_name, args)))
        return self.remotes[0].recv()
