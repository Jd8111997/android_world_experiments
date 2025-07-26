#!/usr/bin/env python3
"""
Integration module to connect T3A enhancements with the original framework
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from framework import AndroidWorldFramework
from t3a_evaluation_framework import T3AEvaluationFramework, T3AEpisodeResult, T3ABenchmarkResults
from enhanced_t3a_agent import EnhancedT3A
from android_world.agents import infer

@dataclass
class T3AComparisonResult:
    """Results comparing T3A approaches with original framework"""
    original_framework_results: List[Dict]
    t3a_enhanced_results: List[T3AEpisodeResult]
    performance_comparison: Dict[str, Any]
    reasoning_analysis: Dict[str, Any]
    recommendations: List[str]

class T3AFrameworkIntegration:
    """Integration layer between T3A enhancements and original framework"""
    
    def __init__(self):
        self.android_framework = None
        self.t3a_evaluator = None
        self.original_evaluator = None
        
    def setup_integrated_environment(self) -> bool:
        """Setup both original and T3A evaluation environments"""
        try:
            # Setup Android World framework
            self.android_framework = AndroidWorldFramework(emulator_setup=False)
            if not self.android_framework.setup_environment():
                return False
            
            # Setup T3A evaluator with corrected LLM wrapper creation
            self.t3a_evaluator = T3AEvaluationFramework(self.android_framework)
            
            # Fix the LLM wrapper creation method
            def create_proper_llm_wrapper(provider: str = "openai", model: str = None):
                if provider == "openai":
                    model = model or "gpt-4o-mini"
                    return infer.Gpt4Wrapper(model)
                elif provider == "anthropic":
                    model = model or "claude-3-sonnet-20240229"
                    return infer.ClaudeWrapper(model)
                else:
                    raise ValueError(f"Unsupported provider: {provider}")
            
            # Replace the method with correct implementation
            self.t3a_evaluator.create_llm_wrapper = create_proper_llm_wrapper
            
            # Setup original evaluator
            from evaluation import EvaluationFramework
            self.original_evaluator = EvaluationFramework(self.android_framework)
            
            print("✅ Integrated environment setup successful")
            return True
            
        except Exception as e:
            print(f"❌ Integrated environment setup failed: {e}")
            logging.error(f"Environment setup failed: {e}")
            return False
    
    def run_comparative_evaluation(self, 
                                 num_episodes: int = 8,
                                 test_tasks: List[str] = None) -> T3AComparisonResult:
        """Run comparative evaluation between original framework and enhanced T3A"""
        
        print(f"\n🔬 Comparative Evaluation: Original Framework vs Enhanced T3A")
        print(f"Episodes: {num_episodes}")
        print("=" * 70)
        
        # Get test tasks
        if test_tasks is None:
            available_tasks = self.android_framework.list_available_tasks()
            test_tasks = available_tasks[:num_episodes] if len(available_tasks) >= num_episodes else available_tasks
        
        original_results = []
        t3a_results = []
        
        for i, task_name in enumerate(test_tasks, 1):
            print(f"\n📱 Task {i}/{len(test_tasks)}: {task_name}")
            
            # Test original framework (best template)
            print("  Testing: Original Framework (few_shot)")
            try:
                orig_result = self.original_evaluator.run_single_episode_evaluation(
                    task_name=task_name,
                    llm_provider="openai",
                    template_name="few_shot",
                    max_steps=20
                )
                original_results.append({
                    'task_name': orig_result.task_name,
                    'success': orig_result.success,
                    'steps_taken': orig_result.steps_taken,
                    'step_accuracy': orig_result.step_accuracy,
                    'approach': 'original_framework'
                })
            except Exception as e:
                print(f"    ❌ Original framework failed: {e}")
            
            # Test enhanced T3A (best approach)
            print("  Testing: Enhanced T3A (chain_of_thought)")
            try:
                t3a_result = self.t3a_evaluator.run_single_t3a_episode(
                    task_name=task_name,
                    prompt_type="chain_of_thought",
                    llm_provider="openai",
                    max_steps=20
                )
                t3a_results.append(t3a_result)
            except Exception as e:
                print(f"    ❌ Enhanced T3A failed: {e}")
        
        # Calculate comparison metrics
        comparison = self._calculate_comparison_metrics(original_results, t3a_results)
        
        # Analyze reasoning quality
        reasoning_analysis = self._analyze_reasoning_differences(original_results, t3a_results)
        
        # Generate recommendations
        recommendations = self._generate_integration_recommendations(comparison, reasoning_analysis)
        
        result = T3AComparisonResult(
            original_framework_results=original_results,
            t3a_enhanced_results=t3a_results,
            performance_comparison=comparison,
            reasoning_analysis=reasoning_analysis,
            recommendations=recommendations
        )
        
        self._print_comparison_summary(result)
        
        return result
    
    def run_t3a_prompt_optimization(self, 
                                   target_tasks: List[str] = None,
                                   optimization_rounds: int = 3) -> Dict[str, Any]:
        """Optimize T3A prompts based on task performance"""
        
        print(f"\n🔧 T3A Prompt Optimization")
        print(f"Optimization Rounds: {optimization_rounds}")
        print("=" * 50)
        
        if target_tasks is None:
            available_tasks = self.android_framework.list_available_tasks()
            target_tasks = available_tasks[:5]  # Test on 5 tasks
        
        optimization_results = {
            'rounds': [],
            'best_configuration': None,
            'improvement_metrics': {},
            'final_recommendations': []
        }
        
        current_best_score = 0
        current_best_config = None
        
        for round_num in range(1, optimization_rounds + 1):
            print(f"\n--- Optimization Round {round_num} ---")
            
            # Test different configurations this round
            if round_num == 1:
                # Baseline comparison
                test_configs = ["original", "chain_of_thought"]
            elif round_num == 2:
                # Add self-reflection
                test_configs = ["chain_of_thought", "self_reflection", "combined"]
            else:
                # Fine-tune best performing
                test_configs = ["combined"]  # Focus on best approach
            
            round_results = []
            
            for config in test_configs:
                print(f"  Testing configuration: {config}")
                
                config_score = 0
                config_details = []
                
                for task_name in target_tasks:
                    try:
                        result = self.t3a_evaluator.run_single_t3a_episode(
                            task_name=task_name,
                            prompt_type=config,
                            llm_provider="openai",
                            max_steps=15
                        )
                        
                        # Calculate weighted score
                        success_weight = 0.4
                        efficiency_weight = 0.3  # Fewer steps better
                        reasoning_weight = 0.3
                        
                        success_score = 1.0 if result.success else 0.0
                        efficiency_score = max(0, 1.0 - (result.steps_taken / 20))  # Normalize to 20 steps
                        reasoning_score = result.reasoning_quality.get('quality_score', 0)
                        
                        task_score = (success_score * success_weight + 
                                    efficiency_score * efficiency_weight + 
                                    reasoning_score * reasoning_weight)
                        
                        config_score += task_score
                        config_details.append({
                            'task': task_name,
                            'success': result.success,
                            'steps': result.steps_taken,
                            'reasoning_quality': reasoning_score,
                            'score': task_score
                        })
                        
                    except Exception as e:
                        print(f"    ❌ Task {task_name} failed: {e}")
                
                avg_score = config_score / len(target_tasks)
                round_results.append({
                    'config': config,
                    'avg_score': avg_score,
                    'details': config_details
                })
                
                print(f"    Average Score: {avg_score:.3f}")
                
                # Update best configuration
                if avg_score > current_best_score:
                    current_best_score = avg_score
                    current_best_config = config
                    print(f"    🎉 New best configuration!")
            
            optimization_results['rounds'].append({
                'round': round_num,
                'configurations_tested': test_configs,
                'results': round_results,
                'best_this_round': max(round_results, key=lambda x: x['avg_score'])
            })
        
        optimization_results['best_configuration'] = current_best_config
        optimization_results['best_score'] = current_best_score
        
        print(f"\n🏆 Optimization Complete")
        print(f"Best Configuration: {current_best_config}")
        print(f"Best Score: {current_best_score:.3f}")
        
        return optimization_results
    
    def _calculate_comparison_metrics(self, original_results: List[Dict], 
                                    t3a_results: List[T3AEpisodeResult]) -> Dict[str, Any]:
        """Calculate comparison metrics between approaches"""
        
        if not original_results or not t3a_results:
            return {}
        
        # Align results by task name for fair comparison
        aligned_comparisons = []
        
        for orig in original_results:
            matching_t3a = next((t3a for t3a in t3a_results if t3a.task_name == orig['task_name']), None)
            if matching_t3a:
                aligned_comparisons.append({
                    'task': orig['task_name'],
                    'original_success': orig['success'],
                    'original_steps': orig['steps_taken'],
                    'original_accuracy': orig['step_accuracy'],
                    't3a_success': matching_t3a.success,
                    't3a_steps': matching_t3a.steps_taken,
                    't3a_accuracy': matching_t3a.step_success_rate,
                    't3a_reasoning_quality': matching_t3a.reasoning_quality.get('quality_score', 0)
                })
        
        if not aligned_comparisons:
            return {}
        
        # Calculate aggregate metrics
        orig_success_rate = sum(1 for c in aligned_comparisons if c['original_success']) / len(aligned_comparisons)
        t3a_success_rate = sum(1 for c in aligned_comparisons if c['t3a_success']) / len(aligned_comparisons)
        
        orig_avg_steps = sum(c['original_steps'] for c in aligned_comparisons) / len(aligned_comparisons)
        t3a_avg_steps = sum(c['t3a_steps'] for c in aligned_comparisons) / len(aligned_comparisons)
        
        orig_avg_accuracy = sum(c['original_accuracy'] for c in aligned_comparisons) / len(aligned_comparisons)
        t3a_avg_accuracy = sum(c['t3a_accuracy'] for c in aligned_comparisons) / len(aligned_comparisons)
        
        avg_reasoning_quality = sum(c['t3a_reasoning_quality'] for c in aligned_comparisons) / len(aligned_comparisons)
        
        return {
            'aligned_comparisons': aligned_comparisons,
            'comparison_count': len(aligned_comparisons),
            'original_framework': {
                'success_rate': orig_success_rate,
                'avg_steps': orig_avg_steps,
                'avg_accuracy': orig_avg_accuracy
            },
            'enhanced_t3a': {
                'success_rate': t3a_success_rate,
                'avg_steps': t3a_avg_steps,
                'avg_accuracy': t3a_avg_accuracy,
                'avg_reasoning_quality': avg_reasoning_quality
            },
            'improvements': {
                'success_rate_diff': t3a_success_rate - orig_success_rate,
                'steps_diff': orig_avg_steps - t3a_avg_steps,  # Positive means T3A uses fewer steps
                'accuracy_diff': t3a_avg_accuracy - orig_avg_accuracy
            }
        }
    
    def _analyze_reasoning_differences(self, original_results: List[Dict], 
                                     t3a_results: List[T3AEpisodeResult]) -> Dict[str, Any]:
        """Analyze reasoning quality differences"""
        
        reasoning_analysis = {
            'reasoning_available': len([r for r in t3a_results if r.reasoning_traces]),
            'avg_reasoning_length': 0,
            'reasoning_completeness': 0,
            'correlation_with_success': 0,
            'reasoning_patterns': {}
        }
        
        if not t3a_results:
            return reasoning_analysis
        
        # Analyze reasoning traces
        all_traces = []
        successful_traces = []
        failed_traces = []
        
        for result in t3a_results:
            for trace in result.reasoning_traces:
                all_traces.append(trace)
                if result.success:
                    successful_traces.append(trace)
                else:
                    failed_traces.append(trace)
        
        if all_traces:
            # Calculate average reasoning characteristics
            avg_length = sum(len(str(trace)) for trace in all_traces) / len(all_traces)
            reasoning_analysis['avg_reasoning_length'] = avg_length
            
            # Analyze completeness (number of filled reasoning components)
            avg_completeness = sum(
                len([k for k, v in trace.items() if v and len(str(v)) > 10]) 
                for trace in all_traces
            ) / len(all_traces)
            reasoning_analysis['reasoning_completeness'] = avg_completeness
            
            # Find patterns in successful vs failed reasoning
            if successful_traces and failed_traces:
                successful_avg_length = sum(len(str(trace)) for trace in successful_traces) / len(successful_traces)
                failed_avg_length = sum(len(str(trace)) for trace in failed_traces) / len(failed_traces)
                
                reasoning_analysis['reasoning_patterns'] = {
                    'successful_reasoning_longer': successful_avg_length > failed_avg_length,
                    'successful_avg_length': successful_avg_length,
                    'failed_avg_length': failed_avg_length,
                    'length_difference': successful_avg_length - failed_avg_length
                }
        
        return reasoning_analysis
    
    def _generate_integration_recommendations(self, comparison: Dict[str, Any], 
                                            reasoning: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison analysis"""
        
        recommendations = []
        
        if not comparison:
            recommendations.append("Insufficient data for comparison analysis")
            return recommendations
        
        # Performance-based recommendations
        improvements = comparison.get('improvements', {})
        
        if improvements.get('success_rate_diff', 0) > 0.1:
            recommendations.append(
                f"Enhanced T3A shows {improvements['success_rate_diff']:.1%} better success rate - "
                "recommend adopting T3A architecture"
            )
        elif improvements.get('success_rate_diff', 0) < -0.1:
            recommendations.append(
                "Original framework outperforms T3A - investigate T3A configuration issues"
            )
        
        if improvements.get('steps_diff', 0) > 2:
            recommendations.append(
                f"T3A is more efficient by {improvements['steps_diff']:.1f} steps on average - "
                "adopt for step-critical applications"
            )
        
        if improvements.get('accuracy_diff', 0) > 0.1:
            recommendations.append(
                f"T3A shows {improvements['accuracy_diff']:.1%} better step accuracy - "
                "improved action reliability"
            )
        
        # Reasoning-based recommendations
        if reasoning.get('avg_reasoning_length', 0) > 200:
            recommendations.append(
                "Rich reasoning traces available - leverage for debugging and improvement"
            )
        
        patterns = reasoning.get('reasoning_patterns', {})
        if patterns.get('successful_reasoning_longer', False):
            length_diff = patterns.get('length_difference', 0)
            recommendations.append(
                f"Longer reasoning correlates with success (+{length_diff:.0f} chars) - "
                "encourage detailed reasoning for complex tasks"
            )
        
        # Integration recommendations
        t3a_quality = comparison.get('enhanced_t3a', {}).get('avg_reasoning_quality', 0)
        if t3a_quality > 0.7:
            recommendations.append(
                "High reasoning quality detected - suitable for production deployment"
            )
        elif t3a_quality > 0.5:
            recommendations.append(
                "Moderate reasoning quality - recommend additional prompt tuning"
            )
        else:
            recommendations.append(
                "Low reasoning quality - investigate prompt design and model capabilities"
            )
        
        return recommendations
    
    def _print_comparison_summary(self, result: T3AComparisonResult):
        """Print comprehensive comparison summary"""
        
        print(f"\n📊 COMPARISON SUMMARY")
        print("=" * 50)
        
        comparison = result.performance_comparison
        
        if comparison:
            print(f"Tasks Compared: {comparison['comparison_count']}")
            
            orig_stats = comparison['original_framework']
            t3a_stats = comparison['enhanced_t3a']
            improvements = comparison['improvements']
            
            print(f"\nPERFORMANCE COMPARISON:")
            print(f"                    Original  │  T3A Enhanced  │  Difference")
            print(f"Success Rate:       {orig_stats['success_rate']:8.1%}  │  {t3a_stats['success_rate']:11.1%}  │  {improvements['success_rate_diff']:+.1%}")
            print(f"Avg Steps:          {orig_stats['avg_steps']:8.1f}  │  {t3a_stats['avg_steps']:11.1f}  │  {improvements['steps_diff']:+.1f}")
            print(f"Step Accuracy:      {orig_stats['avg_accuracy']:8.1%}  │  {t3a_stats['avg_accuracy']:11.1%}  │  {improvements['accuracy_diff']:+.1%}")
            print(f"Reasoning Quality:         -  │  {t3a_stats['avg_reasoning_quality']:11.2f}  │        -")
        
        reasoning = result.reasoning_analysis
        if reasoning.get('reasoning_available', 0) > 0:
            print(f"\nREASONING ANALYSIS:")
            print(f"Traces Available: {reasoning['reasoning_available']}")
            print(f"Avg Length: {reasoning['avg_reasoning_length']:.0f} characters")
            print(f"Completeness: {reasoning['reasoning_completeness']:.1f} components")
        
        if result.recommendations:
            print(f"\nRECOMMENDATIONS:")
            for i, rec in enumerate(result.recommendations, 1):
                print(f"{i}. {rec}")
    
    def close(self):
        """Clean up resources"""
        if self.android_framework:
            self.android_framework.close()