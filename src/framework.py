#!/usr/bin/env python3
"""
Core Android World Framework Module
"""

import random
import logging
from typing import Optional, List, Tuple
from pathlib import Path

# Android World imports
from android_world import registry
from android_world.env import env_launcher
from android_world.agents import m3a_utils
from android_world.env import representation_utils

def find_adb_directory() -> str:
    """Returns the directory where adb is located."""
    import os
    potential_paths = [
        os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
        os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
    ]
    for path in potential_paths:
        if os.path.isfile(path):
            return path
    raise EnvironmentError(
        'adb not found in the common Android SDK paths. Please install Android'
        " SDK and ensure adb is in one of the expected directories."
    )

def generate_ui_elements_description_list_full(
    ui_elements: list,
    screen_width_height_px: tuple,
) -> str:
    """Generate description for a list of UIElement using T3A's method."""
    tree_info = ''
    for index, ui_element in enumerate(ui_elements):
        if m3a_utils.validate_ui_element(ui_element, screen_width_height_px):
            tree_info += f'UI element {index}: {str(ui_element)}\n'
    return tree_info

class AndroidWorldFramework:
    """Main framework for running Android World evaluations"""
    
    def __init__(self, 
                 adb_path: Optional[str] = None,
                 console_port: int = 5554,
                 emulator_setup: bool = False):
        
        self.adb_path = adb_path or find_adb_directory()
        self.console_port = console_port
        self.emulator_setup = emulator_setup
        self.env = None
        
        # Initialize task registry
        self.task_registry = registry.TaskRegistry()
        self.aw_registry = self.task_registry.get_registry(self.task_registry.ANDROID_WORLD_FAMILY)
        
        # Initialize evaluation framework (will be set in setup_environment)
        self.evaluator = None
    
    def setup_environment(self) -> bool:
        """Initialize Android World environment"""
        try:
            print("🔧 Setting up Android World environment...")
            self.env = env_launcher.load_and_setup_env(
                console_port=self.console_port,
                emulator_setup=self.emulator_setup,
                adb_path=self.adb_path,
            )
            self.env.reset(go_home=True)
            
            # Import here to avoid circular imports
            from evaluation import EvaluationFramework
            self.evaluator = EvaluationFramework(self)
            
            print("✅ Environment setup successful")
            return True
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            logging.error(f"Environment setup failed: {e}")
            return False
    
    def list_available_tasks(self) -> List[str]:
        """List all available tasks in Android World"""
        return list(self.aw_registry.keys())
    
    def load_episode(self, task_name: Optional[str] = None) -> Tuple:
        """Load one episode and show goal + first observation"""
        
        if not self.env:
            raise RuntimeError("Environment not initialized")
        
        # If no task specified, let user choose or pick random
        if task_name is None:
            available_tasks = self.list_available_tasks()
            print(f"\nAvailable Tasks ({len(available_tasks)} total):")
            for i, task in enumerate(available_tasks, 1):
                print(f"  {i}. {task}")
            choice = input(f"\nEnter task name, number (1-{len(available_tasks)}), or press Enter for random: ").strip()
            
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_tasks):
                    task_name = available_tasks[idx]
                else:
                    print("Invalid choice, selecting random task")
                    task_name = random.choice(available_tasks)
            elif choice and choice in available_tasks:
                task_name = choice
            else:
                task_name = random.choice(available_tasks)
                print(f"Selected random task: {task_name}")
        
        if task_name not in self.aw_registry:
            raise ValueError(f'Task {task_name} not found')
        
        task_type = self.aw_registry[task_name]
        params = task_type.generate_random_params()
        task = task_type(params)
        task.initialize_task(self.env)
        
        print(f"\n📱 Episode Loaded")
        print(f"Task: {task_name}")
        print(f"Goal: {task.goal}")
        print(f"Complexity: {getattr(task, 'complexity', 'Unknown')}")
        
        return task, task_name, task_type, params
    
    def create_fresh_task(self, task_type, params):
        """Create a fresh task instance and initialize it"""
        fresh_task = task_type(params)
        fresh_task.initialize_task(self.env)
        return fresh_task
    
    def close(self):
        """Clean up environment"""
        if self.env:
            self.env.close()