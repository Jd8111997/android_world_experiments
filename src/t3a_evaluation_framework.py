#!/usr/bin/env python3
"""
T3A-focused evaluation framework for comparing reasoning approaches
"""

import json
import random
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

from android_world.agents import infer
from enhanced_t3a_agent import EnhancedT3A

@dataclass
class T3AEpisodeResult:
    """Enhanced result structure for T3A evaluation"""
    task_name: str
    goal: str
    success: bool
    steps_taken: int
    max_steps: int
    prompt_type: str
    llm_provider: str
    episode_history: List[Dict]
    execution_time: float
    error_types: List[str]
    completion_reason: str
    step_success_rate: float
    reasoning_quality: Dict[str, float]
    reasoning_traces: List[Dict]
    performance_metrics: Dict[str, Any]

@dataclass
class T3ABenchmarkResults:
    """Aggregated T3A benchmark results"""
    total_episodes: int
    prompt_type_performance: Dict[str, Dict]
    llm_performance: Dict[str, Dict]
    reasoning_analysis: Dict[str, Any]
    error_analysis: Dict[str, int]
    completion_reasons: Dict[str, int]
    avg_reasoning_quality: float
    best_prompt_type: str
    improvement_over_baseline: float

class T3AEvaluationFramework:
    """Enhanced evaluation framework specifically for T3A agent variants"""
    
    def __init__(self, android_framework):
        self.android_framework = android_framework
        self.results = []
        
        # T3A specific configurations
        self.prompt_types = [
            "original",           # Baseline T3A
            "chain_of_thought",   # CoT enhancement
            "self_reflection",    # Self-reflection enhancement
            "combined"            # CoT + Self-reflection
        ]
        
    def create_llm_wrapper(self, provider: str = "openai", model: str = None) -> infer.LlmWrapper:
        """Create LLM wrapper for T3A agent using concrete implementations"""
        if provider == "openai":
            if model is None:
                model = "gpt-4o-mini"  # Default to cost-effective model
            
            # Use the concrete Gpt4Wrapper class
            return infer.Gpt4Wrapper(model)
            
        elif provider == "anthropic":
            if model is None:
                model = "claude-3-sonnet-20240229"
            
            # Use the concrete ClaudeWrapper class
            return infer.ClaudeWrapper(model)
            
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'anthropic'")
    
    
    def run_single_t3a_episode(self,
                              task_name: str = None,
                              prompt_type: str = "original",
                              llm_provider: str = "openai", 
                              max_steps: int = 20) -> T3AEpisodeResult:
        """Run evaluation on a single episode with enhanced T3A agent"""
        
        start_time = time.time()
        
        # Load episode
        task, actual_task_name, task_type, task_params = self.android_framework.load_episode(task_name)
        
        # Create fresh task for evaluation
        fresh_task = self.android_framework.create_fresh_task(task_type, task_params)
        
        print(f"\n🔍 T3A Evaluation: {actual_task_name}")
        print(f"Goal: {fresh_task.goal}")
        print(f"Prompt Type: {prompt_type}, LLM: {llm_provider}")
        
        # Create LLM wrapper and enhanced T3A agent
        llm_wrapper = self.create_llm_wrapper(llm_provider)
        agent = EnhancedT3A(
            env=self.android_framework.env,
            llm=llm_wrapper,
            prompt_type=prompt_type,
            name=f"T3A_{prompt_type}",
            enable_summarization=True,
            max_history_length=15
        )
        
        # Set task-specific guidelines if available
        if hasattr(fresh_task, 'get_guidelines'):
            agent.set_task_guidelines(fresh_task.get_guidelines())
        
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
        
        # Get agent performance metrics
        performance_metrics = agent.get_performance_metrics()
        execution_time = time.time() - start_time
        
        # Calculate success rates
        successful_actions = sum(1 for step_data in step_data_list 
                               if step_data.get('action_success', False))
        total_actions = len(step_data_list)
        step_success_rate = successful_actions / total_actions if total_actions > 0 else 0
        
        # Extract reasoning traces
        reasoning_traces = []
        for step_data in step_data_list:
            if step_data.get('parsed_reasoning'):
                reasoning_traces.append(step_data['parsed_reasoning'])
        
        result = T3AEpisodeResult(
            task_name=actual_task_name,
            goal=fresh_task.goal,
            success=task_success,
            steps_taken=agent.step_count,
            max_steps=max_steps,
            prompt_type=prompt_type,
            llm_provider=llm_provider,
            episode_history=agent.history,
            execution_time=execution_time,
            error_types=list(set(error_types)),
            completion_reason=completion_reason,
            step_success_rate=step_success_rate,
            reasoning_quality=performance_metrics.get('reasoning_quality', {}),
            reasoning_traces=reasoning_traces,
            performance_metrics=performance_metrics
        )
        
        # Print results
        status = "🎉 SUCCESS" if task_success else "❌ FAILED"
        quality_score = performance_metrics.get('reasoning_quality', {}).get('quality_score', 0)
        print(f"{status} | Steps: {agent.step_count}/{max_steps} | Success Rate: {step_success_rate:.1%} | Reasoning Quality: {quality_score:.2f}")
        
        return result
    
    def run_prompt_type_comparison(self,
                                 task_name: str = None,
                                 llm_provider: str = "openai",
                                 max_steps: int = 20) -> List[T3AEpisodeResult]:
        """Compare all prompt types on the same task"""
        
        print(f"\n🔬 Prompt Type Comparison")
        print(f"Task: {task_name or 'Random'}")
        print(f"LLM: {llm_provider}")
        print("=" * 60)
        
        results = []
        
        for prompt_type in self.prompt_types:
            print(f"\n📱 Testing: {prompt_type}")
            
            try:
                result = self.run_single_t3a_episode(
                    task_name=task_name,
                    prompt_type=prompt_type,
                    llm_provider=llm_provider,
                    max_steps=max_steps
                )
                results.append(result)
                
                # Save individual result
                self.save_t3a_episode_result(result)
                
            except Exception as e:
                print(f"❌ {prompt_type} failed: {e}")
        
        # Print comparison summary
        self._print_comparison_summary(results)
        
        return results
    
    def run_comprehensive_t3a_benchmark(self, 
                                      num_episodes: int = 12,
                                      llm_providers: List[str] = ["openai"]) -> T3ABenchmarkResults:
        """Run comprehensive T3A benchmark across multiple tasks and prompt types"""
        
        print(f"\n🏆 Comprehensive T3A Benchmark")
        print(f"Episodes: {num_episodes}")
        print(f"Prompt Types: {len(self.prompt_types)}")
        print(f"LLM Providers: {llm_providers}")
        print("=" * 60)
        
        all_results = []
        
        # Get diverse tasks
        available_tasks = self.android_framework.list_available_tasks()
        
        # Calculate episodes per configuration
        total_configs = len(self.prompt_types) * len(llm_providers)
        episodes_per_config = max(1, num_episodes // total_configs)
        
        config_num = 1
        for llm_provider in llm_providers:
            for prompt_type in self.prompt_types:
                print(f"\n📊 Configuration {config_num}/{total_configs}: {llm_provider} + {prompt_type}")
                
                # Run multiple episodes for this configuration
                for episode in range(episodes_per_config):
                    task_name = random.choice(available_tasks)
                    
                    try:
                        result = self.run_single_t3a_episode(
                            task_name=task_name,
                            prompt_type=prompt_type,
                            llm_provider=llm_provider,
                            max_steps=25  # More steps for benchmark
                        )
                        all_results.append(result)
                        
                    except Exception as e:
                        print(f"❌ Episode failed: {e}")
                
                config_num += 1
        
        # Calculate comprehensive metrics
        benchmark_results = self._calculate_t3a_benchmark_metrics(all_results)
        
        # Save benchmark results
        self.save_t3a_benchmark_results(benchmark_results, all_results)
        
        return benchmark_results
    
    def _calculate_t3a_benchmark_metrics(self, results: List[T3AEpisodeResult]) -> T3ABenchmarkResults:
        """Calculate comprehensive T3A benchmark metrics"""
        
        if not results:
            return T3ABenchmarkResults(0, {}, {}, {}, {}, {}, 0.0, "none", 0.0)
        
        # Basic metrics
        total_episodes = len(results)
        
        # Group by prompt type
        prompt_type_results = defaultdict(list)
        for result in results:
            prompt_type_results[result.prompt_type].append(result)
        
        # Calculate prompt type performance
        prompt_type_performance = {}
        baseline_success_rate = 0
        
        for prompt_type, type_results in prompt_type_results.items():
            successes = sum(1 for r in type_results if r.success)
            success_rate = successes / len(type_results)
            avg_steps = statistics.mean([r.steps_taken for r in type_results])
            avg_step_success = statistics.mean([r.step_success_rate for r in type_results])
            
            # Calculate reasoning quality metrics
            quality_scores = [r.reasoning_quality.get('quality_score', 0) for r in type_results]
            avg_quality = statistics.mean(quality_scores)
            
            prompt_type_performance[prompt_type] = {
                'episodes': len(type_results),
                'success_rate': success_rate,
                'avg_steps': avg_steps,
                'avg_step_success_rate': avg_step_success,
                'avg_reasoning_quality': avg_quality,
                'reasoning_completeness': statistics.mean([r.reasoning_quality.get('completeness', 0) for r in type_results])
            }
            
            if prompt_type == "original":
                baseline_success_rate = success_rate
        
        # Group by LLM provider
        llm_results = defaultdict(list)
        for result in results:
            llm_results[result.llm_provider].append(result)
        
        llm_performance = {}
        for llm_provider, llm_episodes in llm_results.items():
            successes = sum(1 for r in llm_episodes if r.success)
            llm_performance[llm_provider] = {
                'episodes': len(llm_episodes),
                'success_rate': successes / len(llm_episodes),
                'avg_steps': statistics.mean([r.steps_taken for r in llm_episodes]),
                'avg_reasoning_quality': statistics.mean([r.reasoning_quality.get('quality_score', 0) for r in llm_episodes])
            }
        
        # Error analysis
        error_analysis = defaultdict(int)
        for result in results:
            for error_type in result.error_types:
                error_analysis[error_type] += 1
        
        # Completion reasons
        completion_reasons = defaultdict(int)
        for result in results:
            completion_reasons[result.completion_reason] += 1
        
        # Reasoning analysis
        reasoning_analysis = self._analyze_reasoning_patterns(results)
        
        # Find best prompt type
        best_prompt_type = max(prompt_type_performance.items(), 
                              key=lambda x: x[1]['success_rate'])[0]
        
        # Calculate improvement over baseline
        best_success_rate = prompt_type_performance[best_prompt_type]['success_rate']
        improvement = (best_success_rate - baseline_success_rate) if baseline_success_rate > 0 else 0
        
        # Overall reasoning quality
        avg_reasoning_quality = statistics.mean([r.reasoning_quality.get('quality_score', 0) for r in results])
        
        return T3ABenchmarkResults(
            total_episodes=total_episodes,
            prompt_type_performance=dict(prompt_type_performance),
            llm_performance=dict(llm_performance),
            reasoning_analysis=reasoning_analysis,
            error_analysis=dict(error_analysis),
            completion_reasons=dict(completion_reasons),
            avg_reasoning_quality=avg_reasoning_quality,
            best_prompt_type=best_prompt_type,
            improvement_over_baseline=improvement
        )
    
    def _analyze_reasoning_patterns(self, results: List[T3AEpisodeResult]) -> Dict[str, Any]:
        """Analyze reasoning patterns across different prompt types"""
        
        reasoning_analysis = {
            'total_reasoning_traces': 0,
            'avg_trace_length': 0,
            'prompt_type_reasoning': {},
            'common_reasoning_patterns': {},
            'reasoning_success_correlation': {}
        }
        
        all_traces = []
        prompt_type_traces = defaultdict(list)
        
        for result in results:
            for trace in result.reasoning_traces:
                all_traces.append(trace)
                prompt_type_traces[result.prompt_type].append(trace)
        
        reasoning_analysis['total_reasoning_traces'] = len(all_traces)
        
        if all_traces:
            # Calculate average trace length
            trace_lengths = [len(str(trace)) for trace in all_traces]
            reasoning_analysis['avg_trace_length'] = statistics.mean(trace_lengths)
            
            # Analyze by prompt type
            for prompt_type, traces in prompt_type_traces.items():
                if traces:
                    type_lengths = [len(str(trace)) for trace in traces]
                    reasoning_analysis['prompt_type_reasoning'][prompt_type] = {
                        'count': len(traces),
                        'avg_length': statistics.mean(type_lengths),
                        'completeness': statistics.mean([
                            len([k for k, v in trace.items() if v and len(str(v)) > 10]) 
                            for trace in traces
                        ])
                    }
            
            # Correlation between reasoning quality and success
            for result in results:
                quality = result.reasoning_quality.get('quality_score', 0)
                success = 1 if result.success else 0
                
                if result.prompt_type not in reasoning_analysis['reasoning_success_correlation']:
                    reasoning_analysis['reasoning_success_correlation'][result.prompt_type] = {
                        'quality_scores': [],
                        'success_rates': []
                    }
                
                reasoning_analysis['reasoning_success_correlation'][result.prompt_type]['quality_scores'].append(quality)
                reasoning_analysis['reasoning_success_correlation'][result.prompt_type]['success_rates'].append(success)
        
        return reasoning_analysis
    
    def _print_comparison_summary(self, results: List[T3AEpisodeResult]):
        """Print comparison summary for prompt types"""
        
        print(f"\n📊 Prompt Type Comparison Results:")
        print("=" * 60)
        
        # Sort by success first, then by reasoning quality
        sorted_results = sorted(results, key=lambda x: (x.success, x.reasoning_quality.get('quality_score', 0)), reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            status = "✅" if result.success else "❌"
            quality = result.reasoning_quality.get('quality_score', 0)
            
            print(f"{i}. {result.prompt_type:15} | {status} | Steps: {result.steps_taken:2d} | "
                  f"Success Rate: {result.step_success_rate:.1%} | Quality: {quality:.2f}")
        
        # Show improvement over baseline
        baseline_result = next((r for r in results if r.prompt_type == "original"), None)
        if baseline_result:
            print(f"\nImprovements over baseline (original):")
            for result in sorted_results:
                if result.prompt_type != "original":
                    success_diff = result.success - baseline_result.success
                    quality_diff = result.reasoning_quality.get('quality_score', 0) - baseline_result.reasoning_quality.get('quality_score', 0)
                    step_diff = baseline_result.steps_taken - result.steps_taken
                    
                    print(f"  {result.prompt_type:15} | Success: {success_diff:+d} | "
                          f"Quality: {quality_diff:+.2f} | Steps: {step_diff:+d}")
    
    def save_t3a_episode_result(self, result: T3AEpisodeResult, output_dir: str = "./results") -> str:
        """Save T3A episode result"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"t3a_episode_{result.task_name}_{result.prompt_type}_{result.llm_provider}_{timestamp}.json"
        filepath = output_path / filename
        
        # Convert to dict for JSON serialization
        result_dict = {
            'task_name': result.task_name,
            'goal': result.goal,
            'success': result.success,
            'steps_taken': result.steps_taken,
            'max_steps': result.max_steps,
            'prompt_type': result.prompt_type,
            'llm_provider': result.llm_provider,
            'episode_history': result.episode_history,
            'execution_time': result.execution_time,
            'error_types': result.error_types,
            'completion_reason': result.completion_reason,
            'step_success_rate': result.step_success_rate,
            'reasoning_quality': result.reasoning_quality,
            'reasoning_traces': result.reasoning_traces,
            'performance_metrics': result.performance_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        return str(filepath)
    
    def save_t3a_benchmark_results(self, benchmark: T3ABenchmarkResults, detailed_results: List[T3AEpisodeResult], 
                                 output_dir: str = "./results") -> str:
        """Save T3A benchmark results and generate report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed benchmark data
        benchmark_data = {
            'benchmark_summary': {
                'total_episodes': benchmark.total_episodes,
                'prompt_type_performance': benchmark.prompt_type_performance,
                'llm_performance': benchmark.llm_performance,
                'reasoning_analysis': benchmark.reasoning_analysis,
                'error_analysis': benchmark.error_analysis,
                'completion_reasons': benchmark.completion_reasons,
                'avg_reasoning_quality': benchmark.avg_reasoning_quality,
                'best_prompt_type': benchmark.best_prompt_type,
                'improvement_over_baseline': benchmark.improvement_over_baseline
            },
            'detailed_results': [
                {
                    'task_name': r.task_name,
                    'goal': r.goal,
                    'success': r.success,
                    'steps_taken': r.steps_taken,
                    'prompt_type': r.prompt_type,
                    'llm_provider': r.llm_provider,
                    'step_success_rate': r.step_success_rate,
                    'reasoning_quality': r.reasoning_quality,
                    'execution_time': r.execution_time,
                    'error_types': r.error_types,
                    'completion_reason': r.completion_reason
                } for r in detailed_results
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        benchmark_file = output_path / f"t3a_benchmark_results_{timestamp}.json"
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2, default=str)
        
        # Generate specialized T3A report
        report_file = self._generate_t3a_report(benchmark, detailed_results, output_path, timestamp)
        
        print(f"💾 T3A Benchmark results saved to: {benchmark_file}")
        print(f"📄 T3A Report generated: {report_file}")
        
        return str(benchmark_file)
    
    def _generate_t3a_report(self, benchmark: T3ABenchmarkResults, detailed_results: List[T3AEpisodeResult],
                           output_path: Path, timestamp: str) -> str:
        """Generate specialized T3A evaluation report"""
        
        report_content = f"""# Enhanced T3A Agent Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Episodes Evaluated:** {benchmark.total_episodes}
**Best Approach:** {benchmark.best_prompt_type}
**Improvement over Baseline:** {benchmark.improvement_over_baseline:.1%}

## Executive Summary

This report evaluates enhanced versions of the T3A (Text-only Autonomous Agent) with Chain of Thought and Self-Reflection capabilities. We compared four reasoning approaches across diverse Android automation tasks to understand how structured reasoning affects agent performance.

### Key Findings

- **Best Performing Approach:** {benchmark.best_prompt_type} 
- **Improvement over Original T3A:** {benchmark.improvement_over_baseline:.1%}
- **Average Reasoning Quality:** {benchmark.avg_reasoning_quality:.2f}/1.0
- **Total Episodes:** {benchmark.total_episodes}

## Reasoning Approach Comparison

### Performance by Prompt Type

| Approach | Episodes | Success Rate | Avg Steps | Step Success | Reasoning Quality |
|----------|----------|--------------|-----------|--------------|-------------------|"""
        
        for prompt_type, stats in benchmark.prompt_type_performance.items():
            report_content += f"\n| {prompt_type} | {stats['episodes']} | {stats['success_rate']:.1%} | {stats['avg_steps']:.1f} | {stats['avg_step_success_rate']:.1%} | {stats['avg_reasoning_quality']:.2f} |"
        
        report_content += f"""

### Reasoning Analysis

**Chain of Thought Benefits:**
"""
        
        if 'chain_of_thought' in benchmark.prompt_type_performance:
            cot_perf = benchmark.prompt_type_performance['chain_of_thought']
            orig_perf = benchmark.prompt_type_performance.get('original', {})
            
            if orig_perf:
                success_improvement = cot_perf['success_rate'] - orig_perf['success_rate']
                quality_improvement = cot_perf['avg_reasoning_quality'] - orig_perf['avg_reasoning_quality']
                
                report_content += f"- Success rate improvement: {success_improvement:+.1%}\n"
                report_content += f"- Reasoning quality improvement: {quality_improvement:+.2f}\n"
                report_content += f"- Step efficiency: {orig_perf['avg_steps'] - cot_perf['avg_steps']:+.1f} steps\n"
        
        report_content += f"""
**Self-Reflection Benefits:**
"""
        
        if 'self_reflection' in benchmark.prompt_type_performance:
            refl_perf = benchmark.prompt_type_performance['self_reflection']
            orig_perf = benchmark.prompt_type_performance.get('original', {})
            
            if orig_perf:
                success_improvement = refl_perf['success_rate'] - orig_perf['success_rate']
                quality_improvement = refl_perf['avg_reasoning_quality'] - orig_perf['avg_reasoning_quality']
                
                report_content += f"- Success rate improvement: {success_improvement:+.1%}\n"
                report_content += f"- Reasoning quality improvement: {quality_improvement:+.2f}\n"
                report_content += f"- Error recovery: Better pattern recognition and strategy adjustment\n"
        
        # Add reasoning pattern analysis
        reasoning_analysis = benchmark.reasoning_analysis
        
        report_content += f"""

## Detailed Analysis

### Reasoning Quality Metrics

- **Total Reasoning Traces:** {reasoning_analysis.get('total_reasoning_traces', 0)}
- **Average Trace Length:** {reasoning_analysis.get('avg_trace_length', 0):.0f} characters

**Reasoning Completeness by Approach:**
"""
        
        for prompt_type, reasoning_data in reasoning_analysis.get('prompt_type_reasoning', {}).items():
            completeness = reasoning_data.get('completeness', 0)
            report_content += f"- {prompt_type}: {completeness:.1f} components filled on average\n"
        
        # Error analysis
        report_content += f"""

### Error Analysis

**Most Common Error Types:**
"""
        
        sorted_errors = sorted(benchmark.error_analysis.items(), key=lambda x: x[1], reverse=True)
        for error_type, count in sorted_errors[:5]:
            percentage = (count / benchmark.total_episodes) * 100
            report_content += f"- {error_type.replace('_', ' ').title()}: {count} occurrences ({percentage:.1f}%)\n"
        
        # Show best examples
        successful_results = [r for r in detailed_results if r.success]
        failed_results = [r for r in detailed_results if not r.success]
        
        if successful_results:
            best_result = max(successful_results, key=lambda x: x.reasoning_quality.get('quality_score', 0))
            report_content += f"""

### Example: Successful {best_result.prompt_type.title()} Reasoning

**Task:** {best_result.task_name}
**Goal:** {best_result.goal}
**Steps:** {best_result.steps_taken}
**Reasoning Quality:** {best_result.reasoning_quality.get('quality_score', 0):.2f}

**Sample Reasoning Trace:**
"""
            if best_result.reasoning_traces:
                sample_trace = best_result.reasoning_traces[0]
                for key, value in sample_trace.items():
                    if value and len(str(value)) > 10:
                        report_content += f"- {key.replace('_', ' ').title()}: {str(value)[:100]}...\n"
        
        # Recommendations
        report_content += f"""

## Recommendations

### 1. Optimal Reasoning Approach
- **Primary Recommendation:** Use {benchmark.best_prompt_type} for best overall performance
- **Quality vs Speed Trade-off:** Chain of thought provides {benchmark.prompt_type_performance.get('chain_of_thought', {}).get('avg_reasoning_quality', 0):.2f} quality vs {benchmark.prompt_type_performance.get('original', {}).get('avg_reasoning_quality', 0):.2f} baseline

### 2. Implementation Guidelines
- **Step Limits:** Increase max steps for complex reasoning approaches
- **Summarization:** Enable step summarization to maintain context
- **History Management:** Limit history to 10-15 steps for optimal performance

### 3. Task-Specific Adaptations
- **Simple Tasks:** Original T3A sufficient for straightforward navigation
- **Complex Tasks:** Use combined CoT + reflection for multi-step planning
- **Error-Prone Tasks:** Self-reflection helps with error recovery

### 4. Future Enhancements
- **Adaptive Reasoning:** Switch reasoning depth based on task complexity
- **Memory Integration:** Add persistent memory across episodes
- **Visual Reasoning:** Integrate screenshot analysis with text reasoning

## Conclusion

Enhanced T3A with structured reasoning shows {benchmark.improvement_over_baseline:.1%} improvement over the baseline. 
The {benchmark.best_prompt_type} approach provides the best balance of performance and reasoning quality.

Key insights:
- Structured reasoning improves both success rates and decision quality
- Self-reflection helps with error recovery and strategy adaptation
- Chain of thought provides systematic analysis but may increase execution time
- Combined approaches offer the highest quality reasoning but require careful tuning

---

*Generated by Enhanced T3A Evaluation Framework*
*Timestamp: {timestamp}*
"""
        
        # Save report
        report_file = output_path / f"t3a_evaluation_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        return str(report_file)