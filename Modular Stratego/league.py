import os
import glob
import random
import shutil
import torch
from typing import Optional, List, Tuple

class LeagueManager:
    """
    Manages a league of historical agents for self-play training.
    Prevents strategy cycling by ensuring the agent plays against diverse past versions of itself.
    """
    
    def __init__(self, league_dir: str = "league_models", max_agents: int = 50):
        """
        Initialize the League Manager.
        
        Args:
            league_dir: Directory to store historical agents
            max_agents: Maximum number of agents to keep in the league
        """
        self.league_dir = league_dir
        self.max_agents = max_agents
        self.agents = []
        
        # Create league directory if it doesn't exist
        os.makedirs(league_dir, exist_ok=True)
        
        # Load existing agents
        self._refresh_agent_list()
        
    def _refresh_agent_list(self):
        """Refresh the list of available agents from disk."""
        self.agents = glob.glob(os.path.join(self.league_dir, "agent_episode_*.pth"))
        self.agents.sort(key=lambda x: int(os.path.basename(x).split('_episode_')[1].split('.')[0]))
        print(f"[INFO] League Manager initialized with {len(self.agents)} historical agents.")
        
    def save_agent(self, agent_path: str, episode: int):
        """
        Save a copy of the current agent to the league.
        
        Args:
            agent_path: Path to the current agent checkpoint
            episode: Current episode number
        """
        if not os.path.exists(agent_path):
            print(f"[WARN] Cannot save to league: Agent file {agent_path} not found.")
            return
            
        league_path = os.path.join(self.league_dir, f"agent_episode_{episode}.pth")
        
        try:
            shutil.copy2(agent_path, league_path)
            print(f"[INFO] Added agent to league: {league_path}")
            self.agents.append(league_path)
            
            # Prune if too many agents (keep recent + random old ones)
            if len(self.agents) > self.max_agents:
                self._prune_league()
                
        except Exception as e:
            print(f"[WARN] Failed to save agent to league: {e}")
            
    def _prune_league(self):
        """Remove excess agents to keep league size manageable."""
        # Strategy: Keep latest 10, random sample of others
        if len(self.agents) <= self.max_agents:
            return
            
        latest = self.agents[-10:]
        others = self.agents[:-10]
        
        # Randomly remove one from others
        to_remove = random.choice(others)
        try:
            os.remove(to_remove)
            self.agents.remove(to_remove)
            print(f"[INFO] Pruned agent from league: {to_remove}")
        except Exception as e:
            print(f"[WARN] Failed to prune agent: {e}")
            
    def get_opponent(self) -> Optional[str]:
        """
        Get a historical opponent from the league.
        
        Returns:
            Path to opponent model file, or None if league is empty
        """
        if not self.agents:
            return None
            
        # Strategy:
        # 50% chance: Latest historical agent (strongest past version)
        # 50% chance: Random historical agent (diversity)
        if random.random() < 0.5:
            return self.agents[-1]
        else:
            return random.choice(self.agents)
