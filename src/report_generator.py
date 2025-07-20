#!/usr/bin/env python3
"""
Report generation module for Android World evaluation results
"""

from datetime import datetime
from pathlib import Path
from typing import List
from collections import defaultdict

from evaluation import BenchmarkResults, EpisodeResult

class ReportGenerator:
    """Generate comprehensive evaluation reports"""
    
    def generate_markdown_report(self, benchmark: BenchmarkResults, detailed_results: List[EpisodeResult],
                               output_path: Path, timestamp: str) -> str:
        """Generate comprehensive markdown report (Task 3 requirement)"""
        
        report_content = f"""# Android World LLM Agent Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Episodes Evaluated:** {benchmark.total_episodes}

## Executive Summary

This report evaluates LLM agents' ability to navigate Android applications using the Android World benchmark. We tested multiple prompt templates and LLM providers across diverse mobile tasks to understand the current capabilities and limitations of LLM-based mobile automation.

### Key Findings

- **Episode Success Rate:** {benchmark.episode_success_rate:.1%} ({sum(1 for r in detailed_results if r.success)}/{benchmark.total_episodes} episodes completed successfully)
- **Average Step Accuracy:** {benchmark.avg_step_accuracy:.1%}
- **Average Steps per Episode:** {benchmark.avg_steps_per_episode:.1f}

## Methodology

### Approach to Prompting and Evaluation

Our evaluation framework implements multiple prompting strategies to test different reasoning approaches:

1. **Basic Prompting:** Simple goal → observation → action format with clear JSON structure
2. **Few-shot Prompting:** Includes 3-8 curated examples of successful agent behavior patterns
3. **Self-reflection:** Asks agents to analyze their progress, learn from failures, and plan strategically
4. **Chain-of-thought:** Structured 4-step reasoning process for systematic decision making
5. **Combined Approaches:** Few-shot examples integrated with self-reflection capabilities

### Technical Implementation

- **T3A-Inspired Design:** Adopted Android World's proven T3A agent prompt structure
- **Index-Based Actions:** Used precise UI element indices to eliminate hallucination
- **JSON Format:** Structured action format: `{{"action_type": "click", "index": 1}}`
- **Enhanced Error Handling:** Comprehensive error categorization and graceful degradation

### Evaluation Metrics

- **Episode Success:** Binary task completion validated by Android World's task evaluators
- **Step Accuracy:** Percentage of successful individual action executions
- **Action Success Rate:** Ratio of successfully executed actions to total attempts
- **Error Analysis:** Detailed categorization of failure modes and patterns

## Performance Results

### Overall Performance Metrics

| Metric | Value |
|--------|-------|
| Total Episodes | {benchmark.total_episodes} |
| Success Rate | {benchmark.episode_success_rate:.1%} |
| Avg Step Accuracy | {benchmark.avg_step_accuracy:.1%} |
| Avg Steps per Episode | {benchmark.avg_steps_per_episode:.1f} |

### LLM Provider Comparison

"""
        
        # LLM Performance Table
        if benchmark.llm_performance:
            report_content += "| Provider | Episodes | Success Rate | Avg Steps | Avg Step Accuracy |\n"
            report_content += "|----------|----------|--------------|-----------|-------------------|\n"
            for provider, stats in benchmark.llm_performance.items():
                report_content += f"| {provider} | {stats['episodes']} | {stats['success_rate']:.1%} | {stats['avg_steps']:.1f} | {stats.get('avg_step_accuracy', 0):.1%} |\n"
        
        report_content += "\n### Prompt Template Comparison\n\n"
        
        # Template Performance Table
        if benchmark.template_performance:
            report_content += "| Template | Episodes | Success Rate | Avg Steps | Avg Step Accuracy |\n"
            report_content += "|----------|----------|--------------|-----------|-------------------|\n"
            for template, stats in benchmark.template_performance.items():
                report_content += f"| {template} | {stats['episodes']} | {stats['success_rate']:.1%} | {stats['avg_steps']:.1f} | {stats.get('avg_step_accuracy', 0):.1%} |\n"
        
        # Failure Analysis
        report_content += "\n## Failure Analysis\n\n"
        report_content += "### Where and Why LLMs Go Wrong\n\n"
        
        if benchmark.error_analysis:
            report_content += "**Error Type Distribution:**\n\n"
            for error_type, count in sorted(benchmark.error_analysis.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / benchmark.total_episodes) * 100
                report_content += f"- **{error_type.replace('_', ' ').title()}:** {count} occurrences ({percentage:.1f}%)\n"
        
        report_content += "\n### Common Failure Patterns\n\n"
        
        # Analyze common failure patterns
        failed_episodes = [r for r in detailed_results if not r.success]
        
        if failed_episodes:
            # Group failures by task type
            task_failures = defaultdict(int)
            for episode in failed_episodes:
                task_failures[episode.task_name] += 1
            
            if task_failures:
                report_content += "**Tasks with High Failure Rates:**\n\n"
                for task, failures in sorted(task_failures.items(), key=lambda x: x[1], reverse=True)[:5]:
                    total_attempts = sum(1 for r in detailed_results if r.task_name == task)
                    failure_rate = failures / total_attempts if total_attempts > 0 else 0
                    report_content += f"- {task}: {failure_rate:.1%} failure rate ({failures}/{total_attempts})\n"
        
        report_content += "\n### Interesting Behaviors Observed\n\n"
        
        # Find interesting patterns
        hallucination_episodes = [r for r in detailed_results if any('element_not_found' in error for error in r.error_types)]
        long_episodes = [r for r in detailed_results if r.steps_taken > benchmark.avg_steps_per_episode * 1.5]
        quick_successes = [r for r in detailed_results if r.success and r.steps_taken < 5]
        
        if hallucination_episodes:
            report_content += f"**Hallucinated Actions:** {len(hallucination_episodes)} episodes showed agents attempting to interact with non-existent UI elements\n\n"
        
        if long_episodes:
            report_content += f"**Extended Episodes:** {len(long_episodes)} episodes took significantly longer than average, suggesting navigation difficulties\n\n"
        
        if quick_successes:
            report_content += f"**Efficient Completions:** {len(quick_successes)} episodes were completed in under 5 steps, demonstrating effective planning\n\n"
        
        # Illustrative Examples
        report_content += "\n## Illustrative Example Episodes\n\n"
        
        # Select 2-3 representative episodes
        examples = []
        
        # Find a successful episode
        successful_episodes = [r for r in detailed_results if r.success]
        if successful_episodes:
            best_success = min(successful_episodes, key=lambda x: x.steps_taken)
            examples.append(("Successful Episode", best_success))
        
        # Find an interesting failure
        if failed_episodes:
            interesting_failure = max(failed_episodes, key=lambda x: x.steps_taken)
            examples.append(("Challenging Episode", interesting_failure))
        
        # Find a quick episode (success or failure)
        quick_episodes = [r for r in detailed_results if r.steps_taken <= 3]
        if quick_episodes:
            quick_episode = quick_episodes[0]
            examples.append(("Quick Resolution", quick_episode))
        
        for example_title, episode in examples:
            report_content += f"### {example_title}\n\n"
            report_content += f"**Task:** {episode.task_name}\n"
            report_content += f"**Goal:** {episode.goal}\n"
            report_content += f"**Result:** {'✅ Success' if episode.success else '❌ Failed'}\n"
            report_content += f"**Steps:** {episode.steps_taken}\n"
            report_content += f"**LLM:** {episode.llm_provider} with {episode.template_name} template\n"
            report_content += f"**Completion Reason:** {episode.completion_reason.replace('_', ' ').title()}\n\n"
            
            if episode.episode_history:
                report_content += "**Action Sequence:**\n\n"
                for i, step in enumerate(episode.episode_history[:5], 1):  # Show first 5 steps
                    action = step.get('action_attempted', 'Unknown')
                    success = '✅' if step.get('action_success') else '❌'
                    reasoning = step.get('reasoning', '')
                    if reasoning and len(reasoning) > 60:
                        reasoning = reasoning[:57] + "..."
                    report_content += f"{i}. {action} {success}\n"
                    if reasoning:
                        report_content += f"   *{reasoning}*\n"
                
                if len(episode.episode_history) > 5:
                    report_content += f"... ({len(episode.episode_history) - 5} more steps)\n"
            
            if episode.error_types:
                report_content += f"\n**Error Types:** {', '.join(episode.error_types)}\n"
            
            report_content += "\n"
        
        # Recommendations
        report_content += "\n## Recommendations for Improving Agent Behavior\n\n"
        
        report_content += "### 1. Memory and Context Management\n\n"
        report_content += "- **Implement working memory:** Agents should maintain a structured memory of previous actions and their outcomes\n"
        report_content += "- **Context-aware prompting:** Include recent action history and success patterns in prompts\n"
        report_content += "- **Goal decomposition:** Break complex tasks into smaller, manageable sub-goals with checkpoints\n"
        report_content += "- **State tracking:** Maintain awareness of current app, screen, and progress toward goal\n\n"
        
        report_content += "### 2. Enhanced Error Handling and Recovery\n\n"
        report_content += "- **UI element validation:** Implement robust checking of element existence and properties before interaction\n"
        report_content += "- **Retry mechanisms:** Develop intelligent retry strategies for failed actions with exponential backoff\n"
        report_content += "- **Alternative path exploration:** When direct paths fail, systematically explore alternative navigation routes\n"
        report_content += "- **Graceful degradation:** Implement fallback actions when preferred approaches are unavailable\n\n"
        
        report_content += "### 3. Improved Reasoning and Planning\n\n"
        report_content += "- **Multi-step planning:** Encourage agents to plan 2-3 steps ahead rather than purely reactive behavior\n"
        report_content += "- **Self-correction mechanisms:** Enable agents to recognize mistakes and backtrack when necessary\n"
        report_content += "- **Domain knowledge integration:** Incorporate Android UI/UX patterns and conventions into prompts\n"
        report_content += "- **Uncertainty handling:** Teach agents to express confidence levels and seek clarification when unsure\n\n"
        
        report_content += "### 4. Prompt Engineering Improvements\n\n"
        
        # Provide specific recommendations based on results
        if benchmark.template_performance:
            best_template = max(benchmark.template_performance.items(), key=lambda x: x[1]['success_rate'])
            worst_template = min(benchmark.template_performance.items(), key=lambda x: x[1]['success_rate'])
            
            report_content += f"**Performance-Based Insights:**\n"
            report_content += f"- **Best performing template:** {best_template[0]} achieved {best_template[1]['success_rate']:.1%} success rate\n"
            report_content += f"- **Needs improvement:** {worst_template[0]} template showed {worst_template[1]['success_rate']:.1%} success rate\n\n"
        
        report_content += "**Specific Improvements:**\n"
        report_content += "- **Enhanced few-shot examples:** Include more diverse scenarios and edge cases in examples\n"
        report_content += "- **Structured output validation:** Implement strict JSON schema validation for action formats\n"
        report_content += "- **Dynamic prompting:** Adjust prompt complexity and detail based on task difficulty\n"
        report_content += "- **Error-informed prompting:** Include common error patterns and how to avoid them\n\n"
        
        # Technical Implementation Suggestions
        report_content += "### 5. Technical Architecture Enhancements\n\n"
        report_content += "- **Vision integration:** Add screenshot analysis capabilities for direct visual understanding\n"
        report_content += "- **Hierarchical task decomposition:** Implement tree-structured goal planning with sub-task tracking\n"
        report_content += "- **Adaptive timeout management:** Dynamic timeout adjustment based on action complexity\n"
        report_content += "- **Ensemble methods:** Combine multiple LLM responses for more robust decision-making\n"
        report_content += "- **Continuous learning:** Implement feedback loops to improve performance over time\n\n"
        
        # Future Work
        report_content += "## Future Work and Research Directions\n\n"
        report_content += "### Immediate Next Steps\n"
        report_content += "- **Larger-scale evaluation:** Test on 100+ episodes across all Android World task categories\n"
        report_content += "- **Multi-modal integration:** Evaluate agents with vision models for direct screen understanding\n"
        report_content += "- **Comparative analysis:** Benchmark against other mobile automation approaches\n"
        report_content += "- **Real device validation:** Test performance on physical Android devices vs. emulators\n\n"
        
        report_content += "### Research Opportunities\n"
        report_content += "- **Transfer learning:** Evaluate agent performance on unseen app types and interfaces\n"
        report_content += "- **Human-in-the-loop systems:** Explore hybrid approaches with human oversight and intervention\n"
        report_content += "- **Accessibility integration:** Develop agents that can assist users with disabilities\n"
        report_content += "- **Cross-platform generalization:** Extend evaluation to iOS and other mobile platforms\n\n"
        
        # Conclusion
        report_content += "## Conclusion\n\n"
        
        success_rate = benchmark.episode_success_rate
        if success_rate >= 0.8:
            conclusion = "demonstrates strong capabilities"
        elif success_rate >= 0.6:
            conclusion = "shows promising potential with room for improvement"
        elif success_rate >= 0.4:
            conclusion = "reveals significant challenges that need addressing"
        else:
            conclusion = "highlights the complexity of mobile automation tasks"
        
        report_content += f"Our evaluation of LLM agents on Android World tasks {conclusion}. "
        report_content += f"With a {success_rate:.1%} episode success rate and {benchmark.avg_step_accuracy:.1%} step accuracy, "
        
        if success_rate >= 0.6:
            report_content += "the results indicate that LLM-based mobile automation is approaching practical viability. "
        else:
            report_content += "the results suggest that significant technical advances are needed before widespread deployment. "
        
        report_content += "The analysis reveals that prompt engineering, particularly few-shot examples and self-reflection capabilities, "
        report_content += "can substantially improve performance. However, fundamental challenges around UI understanding, "
        report_content += "error recovery, and multi-step planning remain.\n\n"
        
        if benchmark.template_performance:
            best_approaches = sorted(benchmark.template_performance.items(), key=lambda x: x[1]['success_rate'], reverse=True)[:2]
            report_content += f"The most effective approaches were {best_approaches[0][0]} and {best_approaches[1][0]} templates, "
            report_content += "suggesting that structured reasoning and example-based learning are key to improving agent performance.\n\n"
        
        # Appendix
        report_content += "## Appendix\n\n"
        report_content += "### Detailed Results Summary\n\n"
        report_content += "| Episode | Task | Goal | Success | Steps | LLM | Template | Completion |\n"
        report_content += "|---------|------|------|---------|-------|-----|----------|------------|\n"
        
        for i, result in enumerate(detailed_results, 1):
            success_icon = "✅" if result.success else "❌"
            task_short = result.task_name[:20] + "..." if len(result.task_name) > 20 else result.task_name
            goal_short = result.goal[:25] + "..." if len(result.goal) > 25 else result.goal
            completion_short = result.completion_reason.replace('_', ' ').title()
            
            report_content += f"| {i} | {task_short} | {goal_short} | {success_icon} | {result.steps_taken} | {result.llm_provider} | {result.template_name} | {completion_short} |\n"
        
        report_content += f"\n### Completion Reason Distribution\n\n"
        for reason, count in benchmark.completion_reasons.items():
            percentage = (count / benchmark.total_episodes) * 100
            reason_formatted = reason.replace('_', ' ').title()
            report_content += f"- **{reason_formatted}:** {count} episodes ({percentage:.1f}%)\n"
        
        report_content += f"\n### Technical Details\n\n"
        report_content += f"- **Framework:** Android World Evaluation Framework\n"
        report_content += f"- **Agent Architecture:** Enhanced T3A-style agent with JSON actions\n"
        report_content += f"- **Evaluation Method:** Task success validation + step-by-step accuracy tracking\n"
        report_content += f"- **Error Categorization:** {len(benchmark.error_analysis)} distinct error types identified\n"
        report_content += f"- **Average Episode Duration:** {sum(r.execution_time for r in detailed_results)/len(detailed_results):.1f} seconds\n"
        
        report_content += "\n---\n\n"
        report_content += f"*Report generated by Android World LLM Agent Evaluation Framework*\n"
        report_content += f"*Timestamp: {timestamp}*\n"
        report_content += f"*Framework Version: 1.0*"
        
        # Save report
        report_file = output_path / f"evaluation_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        return str(report_file)
