#!/usr/bin/env python3
"""
Evaluation framework for Android World LLM agents
"""

import json
import random
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from collections import defaultdict

from llm_client import LLMClient
from prompts import PromptTemplate
from agents import EnhancedAndroidAgent

@dataclass
class EpisodeResult:
    """Structured result for an episode evaluation"""
    task_name: str
    goal: str
    success: bool
    steps_taken: int
    max_steps: int
    action_success_rate: float
    successful_actions: int
    total_actions: int
    llm_provider: str
    template_name: str
    episode_history: List[Dict]
    execution_time: float
    error_types: List[str]
    completion_reason: str
    exact_matches: int
    step_accuracy: float

@dataclass
class BenchmarkResults:
    """Aggregated benchmark results"""
    total_episodes: int
    avg_step_accuracy: float
    episode_success_rate: float
    avg_steps_per_episode: float
    completion_reasons: Dict[str, int]
    error_analysis: Dict[str, int]
    llm_performance: Dict[str, Dict]
    template_performance: Dict[str, Dict]

class EvaluationFramework:
    """Complete evaluation framework for Tasks 2 and 3"""
    
    def __init__(self, android_framework):
        self.android_framework = android_framework
        self.results = []
        
    def run_single_episode_evaluation(self, 
                                    task_name: str = None,
                                    llm_provider: str = "openai",
                                    template_name: str = "basic",
                                    max_steps: int = 15) -> EpisodeResult:
        """Run evaluation on a single episode with detailed metrics"""
        
        start_time = time.time()
        
        # Load episode
        task, actual_task_name, task_type, task_params = self.android_framework.load_episode(task_name)
        
        # Create fresh task for evaluation
        fresh_task = self.android_framework.create_fresh_task(task_type, task_params)
        
        print(f"\n🔍 Evaluating: {actual_task_name}")
        print(f"Goal: {fresh_task.goal}")
        print(f"LLM: {llm_provider}, Template: {template_name}")
        
        # Create components
        llm_client = LLMClient(provider=llm_provider)
        prompt_template = PromptTemplate()
        agent = EnhancedAndroidAgent(
            env=self.android_framework.env, 
            llm_client=llm_client, 
            prompt_template=prompt_template, 
            template_name=template_name,
            name=f"EvalAgent_{template_name}"
        )
        
        # Run episode
        is_done = False
        step_data_list = []
        error_types = []
        completion_reason = "max_steps_reached"
        
        for step in range(max_steps):
            try:
                response = agent.step(fresh_task.goal)
                
                if response.data:
                    step_data_list.append(response.data)
                    
                    # Track error types
                    if response.data.get('error_type'):
                        error_types.append(response.data['error_type'])
                
                if response.done:
                    is_done = True
                    completion_reason = "task_completed"
                    break
                    
            except Exception as e:
                print(f"❌ Step {step + 1} failed: {e}")
                error_types.append("execution_error")
                completion_reason = "execution_error"
                break
        
        # Check task success
        try:
            task_success = is_done and fresh_task.is_successful(self.android_framework.env) == 1
        except Exception as e:
            print(f"⚠️ Could not check task success: {e}")
            task_success = is_done
        
        # Calculate metrics
        successful_actions = sum(1 for step_data in step_data_list 
                               if step_data.get('action_success', False))
        total_actions = len(step_data_list)
        action_success_rate = successful_actions / total_actions if total_actions > 0 else 0
        
        # For step accuracy, we use action success rate as a proxy
        step_accuracy = action_success_rate
        
        execution_time = time.time() - start_time
        
        result = EpisodeResult(
            task_name=actual_task_name,
            goal=fresh_task.goal,
            success=task_success,
            steps_taken=agent.step_count,
            max_steps=max_steps,
            action_success_rate=action_success_rate,
            successful_actions=successful_actions,
            total_actions=total_actions,
            llm_provider=llm_provider,
            template_name=template_name,
            episode_history=agent.episode_history,
            execution_time=execution_time,
            error_types=list(set(error_types)),
            completion_reason=completion_reason,
            exact_matches=successful_actions,  # Proxy for exact matches
            step_accuracy=step_accuracy
        )
        
        # Print results
        status = "🎉 SUCCESS" if task_success else "❌ FAILED"
        print(f"{status} | Steps: {agent.step_count}/{max_steps} | Success Rate: {action_success_rate:.1%}")
        
        return result
    
    def run_multi_episode_evaluation(self, 
                                   num_episodes: int = 3,
                                   llm_providers: List[str] = ["openai"],
                                   template_names: List[str] = ["basic", "few_shot"],
                                   max_steps: int = 15) -> List[EpisodeResult]:
        """Run evaluation on multiple episodes (Task 2 requirement)"""
        
        print(f"\n🔬 Multi-Episode Evaluation")
        print(f"Episodes: {num_episodes}")
        print(f"LLMs: {llm_providers}")
        print(f"Templates: {template_names}")
        print("=" * 60)
        
        all_results = []
        
        # Instead of randomly selecting a task, we are selecting a single task and run num_episodes from it because certain tasks are not compatible with emulator.
        task, task_name, task_type, params = self.android_framework.load_episode()
        selected_tasks = [task_name for _ in range(num_episodes)]
        
        episode_num = 1
        for task_name in selected_tasks:
            for llm_provider in llm_providers:
                for template_name in template_names:
                    print(f"\n📱 Episode {episode_num}/{num_episodes * len(llm_providers) * len(template_names)}")
                    
                    try:
                        result = self.run_single_episode_evaluation(
                            task_name=task_name,
                            llm_provider=llm_provider,
                            template_name=template_name,
                            max_steps=max_steps
                        )
                        all_results.append(result)
                        
                        # Save individual result
                        self.save_episode_result(result)
                        
                    except Exception as e:
                        print(f"❌ Episode failed: {e}")
                    
                    episode_num += 1
        
        self.results.extend(all_results)
        return all_results
    
    def run_benchmark_evaluation(self, num_episodes: int = 10) -> BenchmarkResults:
        """Run comprehensive benchmark evaluation (Task 3 requirement)"""
        
        print(f"\n🏆 Benchmark Evaluation - {num_episodes} Episodes")
        print("=" * 60)
        
        # Test configurations
        configurations = [
            {"llm": "openai", "template": "basic"},
            {"llm": "openai", "template": "few_shot"},
            {"llm": "openai", "template": "self_reflection"},
            {"llm": "openai", "template": "few_shot_reflection"}
        ]
        
        # Add Anthropic if available
        try:
            anthropic_client = LLMClient(provider="anthropic")
            configurations.extend([
                {"llm": "anthropic", "template": "basic"},
                {"llm": "anthropic", "template": "few_shot"}
            ])
        except:
            print("⚠️ Anthropic not available, testing with OpenAI only")
        
        all_results = []
        
        # Instead of randomly selecting a task, we are selecting a single task and run num_episodes from it because certain tasks are not compatible with emulator.
        task, task_name, task_type, params = self.android_framework.load_episode()
        selected_tasks = [task_name for _ in range(num_episodes)]
        
        for i, task_name in enumerate(selected_tasks, 1):
            print(f"\n📊 Benchmark Episode {i}/{num_episodes}: {task_name}")
            
            # Test one configuration per episode to get diverse results
            config = configurations[i % len(configurations)]
            
            try:
                result = self.run_single_episode_evaluation(
                    task_name=task_name,
                    llm_provider=config["llm"],
                    template_name=config["template"],
                    max_steps=20  # More steps for benchmark
                )
                all_results.append(result)
                
            except Exception as e:
                print(f"❌ Benchmark episode failed: {e}")
        
        # Calculate aggregate metrics
        benchmark_results = self._calculate_benchmark_metrics(all_results)
        
        # Save benchmark results
        self.save_benchmark_results(benchmark_results, all_results)
        
        return benchmark_results
    
    def _calculate_benchmark_metrics(self, results: List[EpisodeResult]) -> BenchmarkResults:
        """Calculate aggregated benchmark metrics"""
        
        if not results:
            return BenchmarkResults(0, 0.0, 0.0, 0.0, {}, {}, {}, {})
        
        # Basic metrics
        total_episodes = len(results)
        successful_episodes = sum(1 for r in results if r.success)
        episode_success_rate = successful_episodes / total_episodes
        
        avg_step_accuracy = statistics.mean([r.step_accuracy for r in results])
        avg_steps_per_episode = statistics.mean([r.steps_taken for r in results])
        
        # Completion reasons
        completion_reasons = defaultdict(int)
        for result in results:
            completion_reasons[result.completion_reason] += 1
        
        # Error analysis
        error_analysis = defaultdict(int)
        for result in results:
            for error_type in result.error_types:
                error_analysis[error_type] += 1
        
        # LLM performance comparison
        llm_performance = defaultdict(lambda: {'episodes': 0, 'success_rate': 0.0, 'avg_steps': 0.0})
        llm_stats = defaultdict(list)
        
        for result in results:
            llm_stats[result.llm_provider].append(result)
        
        for llm_provider, llm_results in llm_stats.items():
            successes = sum(1 for r in llm_results if r.success)
            llm_performance[llm_provider] = {
                'episodes': len(llm_results),
                'success_rate': successes / len(llm_results),
                'avg_steps': statistics.mean([r.steps_taken for r in llm_results]),
                'avg_step_accuracy': statistics.mean([r.step_accuracy for r in llm_results])
            }
        
        # Template performance comparison
        template_performance = defaultdict(lambda: {'episodes': 0, 'success_rate': 0.0, 'avg_steps': 0.0})
        template_stats = defaultdict(list)
        
        for result in results:
            template_stats[result.template_name].append(result)
        
        for template_name, template_results in template_stats.items():
            successes = sum(1 for r in template_results if r.success)
            template_performance[template_name] = {
                'episodes': len(template_results),
                'success_rate': successes / len(template_results),
                'avg_steps': statistics.mean([r.steps_taken for r in template_results]),
                'avg_step_accuracy': statistics.mean([r.step_accuracy for r in template_results])
            }
        
        return BenchmarkResults(
            total_episodes=total_episodes,
            avg_step_accuracy=avg_step_accuracy,
            episode_success_rate=episode_success_rate,
            avg_steps_per_episode=avg_steps_per_episode,
            completion_reasons=dict(completion_reasons),
            error_analysis=dict(error_analysis),
            llm_performance=dict(llm_performance),
            template_performance=dict(template_performance)
        )
    
    def save_episode_result(self, result: EpisodeResult, output_dir: str = "./results") -> str:
        """Save individual episode result"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"episode_{result.task_name}_{result.llm_provider}_{result.template_name}_{timestamp}.json"
        filepath = output_path / filename
        
        # Convert to dict for JSON serialization
        result_dict = {
            'task_name': result.task_name,
            'goal': result.goal,
            'success': result.success,
            'steps_taken': result.steps_taken,
            'max_steps': result.max_steps,
            'action_success_rate': result.action_success_rate,
            'successful_actions': result.successful_actions,
            'total_actions': result.total_actions,
            'llm_provider': result.llm_provider,
            'template_name': result.template_name,
            'episode_history': result.episode_history,
            'execution_time': result.execution_time,
            'error_types': result.error_types,
            'completion_reason': result.completion_reason,
            'exact_matches': result.exact_matches,
            'step_accuracy': result.step_accuracy,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        return str(filepath)
    
    def save_benchmark_results(self, benchmark: BenchmarkResults, detailed_results: List[EpisodeResult], 
                             output_dir: str = "./results") -> str:
        """Save benchmark results and generate report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed benchmark data
        benchmark_data = {
            'benchmark_summary': {
                'total_episodes': benchmark.total_episodes,
                'avg_step_accuracy': benchmark.avg_step_accuracy,
                'episode_success_rate': benchmark.episode_success_rate,
                'avg_steps_per_episode': benchmark.avg_steps_per_episode,
                'completion_reasons': benchmark.completion_reasons,
                'error_analysis': benchmark.error_analysis,
                'llm_performance': benchmark.llm_performance,
                'template_performance': benchmark.template_performance
            },
            'detailed_results': [
                {
                    'task_name': r.task_name,
                    'goal': r.goal,
                    'success': r.success,
                    'steps_taken': r.steps_taken,
                    'action_success_rate': r.action_success_rate,
                    'llm_provider': r.llm_provider,
                    'template_name': r.template_name,
                    'execution_time': r.execution_time,
                    'error_types': r.error_types,
                    'completion_reason': r.completion_reason,
                    'step_accuracy': r.step_accuracy
                } for r in detailed_results
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        benchmark_file = output_path / f"benchmark_results_{timestamp}.json"
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2, default=str)
        
        # Generate markdown report
        from report_generator import ReportGenerator
        report_generator = ReportGenerator()
        report_file = report_generator.generate_markdown_report(benchmark, detailed_results, output_path, timestamp)
        
        print(f"💾 Benchmark results saved to: {benchmark_file}")
        print(f"📄 Report generated: {report_file}")
        
        return str(benchmark_file)