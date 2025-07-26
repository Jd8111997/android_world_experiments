#!/usr/bin/env python3
"""
Enhanced T3A Agent with Chain of Thought and Self-Reflection capabilities
"""

import json
import re
from typing import Dict, Any, List, Optional
from collections import defaultdict

from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a_utils
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils

# Import the original T3A prompts and functions
from android_world.agents.t3a import (
    _generate_ui_elements_description_list_full,
    _action_selection_prompt,
    _summarize_prompt,
    ACTION_SELECTION_PROMPT_TEMPLATE,
    SUMMARIZATION_PROMPT_TEMPLATE,
    PROMPT_PREFIX,
    GUIDANCE
)

"""
# Enhanced prompt templates with CoT and Self-Reflection
CHAIN_OF_THOUGHT_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}'
    + '\n\nHere is a history of what you have done so far:\n{history}'
    + '\n\nHere is a list of descriptions for some UI elements on the current'
    ' screen:\n{ui_elements_description}\n'
    + GUIDANCE
    + '{additional_guidelines}'
    + '\n\nNow think through this step by step using chain of thought reasoning:\n'
    + '1. SITUATION ANALYSIS: What do I see on the current screen? What app am I in?\n'
    + '2. GOAL ASSESSMENT: What is my ultimate objective? What progress have I made toward this goal?\n'
    + '3. ACTION PLANNING: What are my available options? Which action best progresses toward the goal?\n'
    + '4. PROGRESS CHECK: Have I completed the goal already? Should I mark it as complete?\n'
    + '5. BACKTRACK EVALUATION: Am I stuck or repeating actions? Should I go back or try a different approach?\n\n'
    + 'IMPORTANT GUIDELINES:\n'
    + '- If you have achieved the goal (e.g., Wi-Fi is already on, app is already uninstalled), use status action with "complete"\n'
    + '- If you are repeating the same action or seem stuck, try navigate_back or a different approach\n'
    + '- If you have tried the same action 2+ times unsuccessfully, change your strategy\n'
    + '- Look for visual confirmation that your goal is achieved before continuing\n\n'
    + 'Provide your step-by-step reasoning, then output an action in the correct JSON format.\n'
    + 'Your answer should look like:\n'
    + 'Situation Analysis: ...\n'
    + 'Goal Assessment: ...\n'
    + 'Action Planning: ...\n'
    + 'Progress Check: ...\n'
    + 'Backtrack Evaluation: ...\n'
    + 'Reason: ...\n'
    + 'Action: {{"action_type":...}}\n\n'
    + 'Your Answer:\n'
)

SELF_REFLECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}'
    + '\n\nHere is a history of what you have done so far:\n{history}'
    + '\n\nHere is a list of descriptions for some UI elements on the current'
    ' screen:\n{ui_elements_description}\n'
    + GUIDANCE
    + '{additional_guidelines}'
    + '\n\nBefore taking action, reflect on your progress and strategy:\n'
    + '1. PROGRESS REFLECTION: How much progress have I made? What has worked well?\n'
    + '2. ERROR ANALYSIS: What mistakes have I made? Am I repeating failed actions?\n'
    + '3. COMPLETION CHECK: Have I already achieved my goal? Should I complete the task?\n'
    + '4. STRATEGY ASSESSMENT: Is my current approach effective? Should I try something different?\n'
    + '5. BACKTRACK DECISION: Am I stuck in a loop? Should I navigate back to try a different path?\n'
    + '6. PATTERN RECOGNITION: Are there UI patterns I should recognize? What do I know about this type of app?\n'
    + '7. NEXT STEP PLANNING: Given my reflection, what is the best next action?\n\n'
    + 'BACKTRACKING LOGIC:\n'
    + '- If you have repeated the same action multiple times, use navigate_back\n'
    + '- If you are in the wrong app or section, navigate back to a previous screen\n'
    + '- If progress has stalled, try a completely different approach\n'
    + '- If the goal appears to be achieved, use status "complete" immediately\n\n'
    + 'Provide your self-reflection, then output an action in the correct JSON format.\n'
    + 'Your answer should look like:\n'
    + 'Progress Reflection: ...\n'
    + 'Error Analysis: ...\n'
    + 'Completion Check: ...\n'
    + 'Strategy Assessment: ...\n'
    + 'Backtrack Decision: ...\n'
    + 'Pattern Recognition: ...\n'
    + 'Next Step Planning: ...\n'
    + 'Reason: ...\n'
    + 'Action: {{"action_type":...}}\n\n'
    + 'Your Answer:\n'
)

COMBINED_COT_REFLECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}'
    + '\n\nHere is a history of what you have done so far:\n{history}'
    + '\n\nHere is a list of descriptions for some UI elements on the current'
    ' screen:\n{ui_elements_description}\n'
    + GUIDANCE
    + '{additional_guidelines}'
    + '\n\nUse both chain of thought reasoning and self-reflection for optimal decision making:\n\n'
    + 'CHAIN OF THOUGHT ANALYSIS:\n'
    + '1. SITUATION: What do I see on the current screen?\n'
    + '2. GOAL: How does this relate to my objective? Am I done?\n'
    + '3. OPTIONS: What actions are available to me?\n'
    + '4. SELECTION: Which action best progresses toward the goal?\n\n'
    + 'SELF-REFLECTION:\n'
    + '1. PROGRESS: How am I doing so far? Any signs the goal is complete?\n'
    + '2. LEARNING: What have I learned from previous actions? Any repeated failures?\n'
    + '3. PATTERNS: What patterns do I recognize in this interface?\n'
    + '4. STRATEGY: Should I continue current approach or backtrack/change strategy?\n\n'
    + 'CRITICAL DECISION POINTS:\n'
    + '- COMPLETION: If goal appears achieved, immediately use {{"action_type": "status", "goal_status": "complete"}}\n'
    + '- BACKTRACKING: If stuck or repeating actions, use {{"action_type": "navigate_back"}}\n'
    + '- STRATEGY CHANGE: If current approach failed 2+ times, try completely different approach\n\n'
    + 'Provide your analysis and reflection, then output an action in the correct JSON format.\n'
    + 'Your answer should look like:\n'
    + 'Chain of Thought - Situation: ...\n'
    + 'Chain of Thought - Goal: ...\n'
    + 'Chain of Thought - Options: ...\n'
    + 'Chain of Thought - Selection: ...\n'
    + 'Self-Reflection - Progress: ...\n'
    + 'Self-Reflection - Learning: ...\n'
    + 'Self-Reflection - Patterns: ...\n'
    + 'Self-Reflection - Strategy: ...\n'
    + 'Reason: ...\n'
    + 'Action: {{"action_type":...}}\n\n'
    + 'Your Answer:\n'
)
"""
CHAIN_OF_THOUGHT_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}'
    + '\n\nHere is a history of what you have done so far:\n{history}'
    + '\n\nHere is a list of descriptions for some UI elements on the current'
    ' screen:\n{ui_elements_description}\n'
    + GUIDANCE
    + '{additional_guidelines}'
    + '\n\nUse systematic reasoning to make the best decision:\n\n'
    + '1. CURRENT SITUATION: What do I see? What app/screen am I on?\n'
    + '2. GOAL PROGRESS: What have I accomplished? What remains to be done?\n'
    + '3. SUCCESS INDICATORS: Are there signs my goal is already complete?\n'
    + '4. ACTION OPTIONS: What specific actions can I take right now?\n'
    + '5. BEST CHOICE: Which action most directly advances my goal?\n\n'
    + 'CRITICAL DECISIONS:\n'
    + '- TASK COMPLETE: If goal is achieved (photo taken, contact added, etc.), immediately use {{"action_type": "status", "goal_status": "complete"}}\n'
    + '- STUCK DETECTION: If I tried same action 2+ times unsuccessfully, use {{"action_type": "navigate_back"}} or try different approach\n'
    + '- FOCUS: Take the most direct action toward the goal, avoid unnecessary exploration\n\n'
    + 'Format your response as:\n'
    + 'Current Situation: [What I see and where I am]\n'
    + 'Goal Progress: [What I\'ve done and what remains]\n'
    + 'Success Check: [Is the goal already complete?]\n'
    + 'Action Options: [Available actions]\n'
    + 'Best Choice: [Selected action and why]\n'
    + 'Reason: [Brief justification]\n'
    + 'Action: {{"action_type":...}}\n\n'
    + 'Your Answer:\n'
)

SELF_REFLECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}'
    + '\n\nHere is a history of what you have done so far:\n{history}'
    + '\n\nHere is a list of descriptions for some UI elements on the current'
    ' screen:\n{ui_elements_description}\n'
    + GUIDANCE
    + '{additional_guidelines}'
    + '\n\nReflect on your progress before acting:\n\n'
    + '1. PROGRESS ANALYSIS: What concrete progress have I made toward the goal?\n'
    + '2. COMPLETION CHECK: Is there evidence the goal is already achieved?\n'
    + '3. ERROR DETECTION: Am I repeating failed actions or stuck in a loop?\n'
    + '4. STRATEGY EVALUATION: Is my current approach working or should I change?\n'
    + '5. NEXT ACTION: What is the most effective next step?\n\n'
    + 'SMART DECISIONS:\n'
    + '- RECOGNIZE SUCCESS: Look for confirmation the task is done (photo saved, contact visible, setting changed)\n'
    + '- AVOID LOOPS: If same action failed twice, try different approach or navigate back\n'
    + '- BE DECISIVE: Take direct action rather than endless checking\n'
    + '- TRUST RESULTS: If action succeeded (no errors), assume it worked\n\n'
    + 'Format your response as:\n'
    + 'Progress Analysis: [Concrete achievements]\n'
    + 'Completion Check: [Evidence goal is done?]\n'
    + 'Error Detection: [Any repeated failures?]\n'
    + 'Strategy Evaluation: [Is approach working?]\n'
    + 'Next Action: [Best next step]\n'
    + 'Reason: [Clear justification]\n'
    + 'Action: {{"action_type":...}}\n\n'
    + 'Your Answer:\n'
)

COMBINED_COT_REFLECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe current user goal/request is: {goal}'
    + '\n\nHere is a history of what you have done so far:\n{history}'
    + '\n\nHere is a list of descriptions for some UI elements on the current'
    ' screen:\n{ui_elements_description}\n'
    + GUIDANCE
    + '{additional_guidelines}'
    + '\n\nCombine systematic thinking with smart reflection:\n\n'
    + 'QUICK ANALYSIS:\n'
    + '1. SITUATION: What screen/app am I in? What do I see?\n'
    + '2. GOAL STATUS: What progress made? Any success indicators?\n'
    + '3. OPTIONS: What actions are available right now?\n\n'
    + 'SMART REFLECTION:\n'
    + '1. SUCCESS SIGNS: Is there evidence my goal is complete?\n'
    + '2. EFFICIENCY: Am I taking the most direct path?\n'
    + '3. LOOP DETECTION: Have I repeated actions without progress?\n\n'
    + 'DECISION RULES:\n'
    + '- COMPLETE IMMEDIATELY: If goal achieved, use status "complete"\n'
    + '- BREAK LOOPS: If stuck repeating actions, navigate back or change approach\n'
    + '- BE DIRECT: Choose actions that directly advance the goal\n'
    + '- TRUST SUCCESS: If actions execute without errors, assume they worked\n\n'
    + 'Analyze the situation using both systematic thinking and reflection, then provide your conclusion:\n'
    + 'Reason: [Provide your final reasoning incorporating both analysis and reflection]\n'
    + 'Action: {{"action_type":...}}\n\n'
    + 'Your Answer:\n'
)

def _enhanced_action_selection_prompt(
    goal: str,
    history: list[str],
    ui_elements_description: str,
    additional_guidelines: list[str] | None = None,
    prompt_type: str = "original"
) -> str:
    """Generate enhanced prompts with CoT and self-reflection capabilities."""
    
    if history:
        history_text = '\n'.join(history)
    else:
        history_text = 'You just started, no action has been performed yet.'

    extra_guidelines = ''
    if additional_guidelines:
        extra_guidelines = 'For The Current Task:\n'
        for guideline in additional_guidelines:
            extra_guidelines += f'- {guideline}\n'

    template_map = {
        "original": ACTION_SELECTION_PROMPT_TEMPLATE,
        "chain_of_thought": CHAIN_OF_THOUGHT_PROMPT_TEMPLATE,
        "self_reflection": SELF_REFLECTION_PROMPT_TEMPLATE,
        "combined": COMBINED_COT_REFLECTION_PROMPT_TEMPLATE
    }
    
    template = template_map.get(prompt_type, ACTION_SELECTION_PROMPT_TEMPLATE)
    
    return template.format(
        history=history_text,
        goal=goal,
        ui_elements_description=ui_elements_description
        if ui_elements_description
        else 'Not available',
        additional_guidelines=extra_guidelines,
    )

class EnhancedT3A(base_agent.EnvironmentInteractingAgent):
    """Enhanced T3A agent with Chain of Thought and Self-Reflection capabilities"""

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.LlmWrapper,
        prompt_type: str = "original",
        name: str = 'Enhanced_T3A',
        enable_summarization: bool = True,
        max_history_length: int = 10,
        backtrack_threshold: int = 3,  # New: threshold for detecting repeated actions
    ):
        """Initialize Enhanced T3A agent.
        
        Args:
            env: The environment.
            llm: The text-only LLM.
            prompt_type: Type of prompting ("original", "chain_of_thought", "self_reflection", "combined")
            name: The agent name.
            enable_summarization: Whether to use step summarization.
            max_history_length: Maximum number of steps to keep in history.
            backtrack_threshold: Number of repeated/similar actions before suggesting backtrack.
        """
        super().__init__(env, name)
        self.llm = llm
        self.prompt_type = prompt_type
        self.enable_summarization = enable_summarization
        self.max_history_length = max_history_length
        self.backtrack_threshold = backtrack_threshold
        self.history = []
        self.additional_guidelines = None
        
        # Enhanced tracking for evaluation
        self.step_count = 0
        self.error_counts = defaultdict(int)
        self.reasoning_traces = []
        self.action_success_history = []
        
        # New: Backtracking and completion detection
        self.recent_actions = []  # Track recent actions for loop detection
        self.completion_indicators = []  # Track potential completion signals
        self.stuck_counter = 0  # Counter for detecting when agent is stuck

    def reset(self, go_home_on_reset: bool = False):
        """Reset agent state."""
        super().reset(go_home_on_reset)
        self.env.hide_automation_ui()
        self.history = []
        self.step_count = 0
        self.error_counts.clear()
        self.reasoning_traces.clear()
        self.action_success_history.clear()
        # Reset backtracking state
        self.recent_actions.clear()
        self.completion_indicators.clear()
        self.stuck_counter = 0

    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        """Set task-specific guidelines."""
        self.additional_guidelines = task_guidelines

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """Take one step with enhanced reasoning capabilities."""
        self.step_count += 1
        
        step_data = {
            'step_number': self.step_count,
            'prompt_type': self.prompt_type,
            'before_screenshot': None,
            'after_screenshot': None,
            'before_element_list': None,
            'after_element_list': None,
            'action_prompt': None,
            'action_output': None,
            'action_raw_response': None,
            'reasoning_trace': None,
            'parsed_reasoning': {},
            'summary_prompt': None,
            'summary': None,
            'summary_raw_response': None,
            'action_success': False,
            'error_type': None,
        }
        
        print(f'----------step {self.step_count} ({self.prompt_type})')

        try:
            # Get current state
            state = self.get_post_transition_state()
            logical_screen_size = self.env.logical_screen_size

            ui_elements = state.ui_elements
            before_element_list = _generate_ui_elements_description_list_full(
                ui_elements, logical_screen_size,
            )
            
            step_data['before_screenshot'] = state.pixels.copy()
            step_data['before_element_list'] = ui_elements

            # Generate enhanced action prompt with backtracking context
            backtrack_context = self._build_backtrack_context()
            completion_context = self._build_completion_context(goal)
            
            action_prompt = _enhanced_action_selection_prompt(
                goal,
                [
                    'Step ' + str(i + 1) + ': ' + step_info['summary']
                    for i, step_info in enumerate(self.history)
                ],
                before_element_list,
                self.additional_guidelines,
                self.prompt_type
            )
            
            # Add backtracking and completion context to prompt
            if backtrack_context or completion_context:
                context_addition = "\n\nIMPORTANT CONTEXT:\n"
                if completion_context:
                    context_addition += completion_context + "\n"
                if backtrack_context:
                    context_addition += backtrack_context + "\n"
                action_prompt += context_addition
            
            step_data['action_prompt'] = action_prompt
            
            # Get LLM response
            action_output, is_safe, raw_response = self.llm.predict(action_prompt)

            if is_safe == False:
                action_output = f"""Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}
Action: {{"action_type": "status", "goal_status": "infeasible"}}"""

            if not raw_response:
                raise RuntimeError('Error calling LLM in action selection phase.')

            step_data['action_output'] = action_output
            step_data['action_raw_response'] = raw_response

            # Parse enhanced reasoning
            reasoning_trace, parsed_reasoning = self._parse_enhanced_reasoning(action_output)
            step_data['reasoning_trace'] = reasoning_trace
            step_data['parsed_reasoning'] = parsed_reasoning
            self.reasoning_traces.append(parsed_reasoning)

            # Parse reason and action (original T3A logic)
            reason, action = m3a_utils.parse_reason_action_output(action_output)

            # Handle parsing failures
            if (not reason) or (not action):
                print('Action prompt output is not in the correct format.')
                step_data['summary'] = (
                    'Output for action selection is not in the correct format, so no'
                    ' action is performed.'
                )
                step_data['error_type'] = 'parsing_error'
                self.error_counts['parsing_error'] += 1
                self.history.append(step_data)
                self.action_success_history.append(False)
                return base_agent.AgentInteractionResult(False, step_data)

            print('Action: ' + action)
            print('Reason: ' + reason)
            if reasoning_trace:
                print('Enhanced Reasoning: ' + reasoning_trace[:200] + '...')

            # Convert action to JSON
            try:
                converted_action = json_action.JSONAction(
                    **m3a_utils.extract_json(action),
                )
            except Exception as e:
                print('Failed to convert the output to a valid action.')
                print(str(e))
                step_data['summary'] = (
                    'Can not parse the output to a valid action. Please make sure to pick'
                    ' the action from the list with the correct json format!'
                )
                step_data['error_type'] = 'json_parsing_error'
                self.error_counts['json_parsing_error'] += 1
                self.history.append(step_data)
                self.action_success_history.append(False)
                return base_agent.AgentInteractionResult(False, step_data)

            # Validate action (original T3A logic)
            if converted_action.action_type in ['click', 'long-press', 'input-text']:
                if converted_action.index is not None and converted_action.index >= len(ui_elements):
                    print('Index out of range.')
                    step_data['summary'] = (
                        'The parameter index is out of range. Remember the index must be in'
                        ' the UI element list!'
                    )
                    step_data['error_type'] = 'index_out_of_range'
                    self.error_counts['index_out_of_range'] += 1
                    self.history.append(step_data)
                    self.action_success_history.append(False)
                    return base_agent.AgentInteractionResult(False, step_data)
                else:
                    # Add mark for visualization
                    m3a_utils.add_ui_element_mark(
                        step_data['before_screenshot'],
                        ui_elements[converted_action.index],
                        converted_action.index,
                        logical_screen_size,
                        adb_utils.get_physical_frame_boundary(self.env.controller),
                        adb_utils.get_orientation(self.env.controller),
                    )

            # Handle completion
            if converted_action.action_type == 'status':
                if converted_action.goal_status == 'infeasible':
                    print('Agent stopped since it thinks mission impossible.')
                step_data['summary'] = 'Agent thinks the request has been completed.'
                step_data['action_success'] = True
                self.action_success_history.append(True)
                self.history.append(step_data)
                return base_agent.AgentInteractionResult(True, step_data)

            if converted_action.action_type == 'answer':
                print('Agent answered with: ' + converted_action.text)

            # Track action for backtracking detection
            self._track_action_for_backtracking(converted_action)
            
            # Check for completion indicators
            self._check_completion_indicators(goal, ui_elements)

            # Execute action
            try:
                self.env.execute_action(converted_action)
                step_data['action_success'] = True
                self.action_success_history.append(True)
                self.stuck_counter = 0  # Reset stuck counter on successful action
            except Exception as e:
                print('Some error happened executing the action', converted_action.action_type)
                print(str(e))
                step_data['summary'] = (
                    'Some error happened executing the action '
                    + converted_action.action_type
                )
                step_data['error_type'] = 'execution_error'
                step_data['action_success'] = False
                self.error_counts['execution_error'] += 1
                self.action_success_history.append(False)
                self.stuck_counter += 1  # Increment stuck counter on failed action
                self.history.append(step_data)
                return base_agent.AgentInteractionResult(False, step_data)

            # Get post-action state
            state = self.get_post_transition_state()
            ui_elements = state.ui_elements
            after_element_list = _generate_ui_elements_description_list_full(
                ui_elements, self.env.logical_screen_size,
            )

            step_data['after_screenshot'] = state.pixels.copy()
            step_data['after_element_list'] = ui_elements

            # Generate summary (if enabled)
            if self.enable_summarization:
                summary = self._generate_enhanced_summary(
                    goal, action, reason, before_element_list, after_element_list, parsed_reasoning
                )
                step_data['summary'] = summary
                print('Summary: ' + summary)
            else:
                step_data['summary'] = f'Action selected: {action}. {reason}'

            # Manage history length
            self.history.append(step_data)
            if len(self.history) > self.max_history_length:
                self.history = self.history[-self.max_history_length:]

            return base_agent.AgentInteractionResult(False, step_data)

        except Exception as e:
            print(f'Error in enhanced T3A step: {e}')
            step_data['summary'] = f'Unexpected error: {str(e)}'
            step_data['error_type'] = 'unexpected_error'
            step_data['action_success'] = False
            self.error_counts['unexpected_error'] += 1
            self.action_success_history.append(False)
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

    def _parse_enhanced_reasoning(self, action_output: str) -> tuple[str, dict]:
        """Parse enhanced reasoning from LLM output."""
        parsed = {}
        reasoning_trace = ""
        
        if self.prompt_type == "chain_of_thought":
            patterns = {
                'situation_analysis': r'Situation Analysis:\s*(.*?)(?=Goal Assessment:|Reason:|$)',
                'goal_assessment': r'Goal Assessment:\s*(.*?)(?=Action Planning:|Reason:|$)',
                'action_planning': r'Action Planning:\s*(.*?)(?=Risk Evaluation:|Reason:|$)',
                'risk_evaluation': r'Risk Evaluation:\s*(.*?)(?=Reason:|$)'
            }
        elif self.prompt_type == "self_reflection":
            patterns = {
                'progress_reflection': r'Progress Reflection:\s*(.*?)(?=Error Analysis:|Reason:|$)',
                'error_analysis': r'Error Analysis:\s*(.*?)(?=Strategy Assessment:|Reason:|$)',
                'strategy_assessment': r'Strategy Assessment:\s*(.*?)(?=Pattern Recognition:|Reason:|$)',
                'pattern_recognition': r'Pattern Recognition:\s*(.*?)(?=Next Step Planning:|Reason:|$)',
                'next_step_planning': r'Next Step Planning:\s*(.*?)(?=Reason:|$)'
            }
        elif self.prompt_type == "combined":
            patterns = {
            'situation': r'Situation:\s*(.*?)(?=Goal Status:|Reason:|Action:|$)',
            'goal_status': r'Goal Status:\s*(.*?)(?=Options:|Reason:|Action:|$)',
            'options': r'Options:\s*(.*?)(?=Reflection:|Reason:|Action:|$)',
            'reflection': r'Reflection:\s*(.*?)(?=Decision:|Reason:|Action:|$)',
            'decision': r'Decision:\s*(.*?)(?=Reason:|Action:|$)'
            }
            """
            patterns = {
                'cot_situation': r'Chain of Thought - Situation:\s*(.*?)(?=Chain of Thought - Goal:|Reason:|$)',
                'cot_goal': r'Chain of Thought - Goal:\s*(.*?)(?=Chain of Thought - Options:|Reason:|$)',
                'cot_options': r'Chain of Thought - Options:\s*(.*?)(?=Chain of Thought - Selection:|Reason:|$)',
                'cot_selection': r'Chain of Thought - Selection:\s*(.*?)(?=Self-Reflection - Progress:|Reason:|$)',
                'reflection_progress': r'Self-Reflection - Progress:\s*(.*?)(?=Self-Reflection - Learning:|Reason:|$)',
                'reflection_learning': r'Self-Reflection - Learning:\s*(.*?)(?=Self-Reflection - Patterns:|Reason:|$)',
                'reflection_patterns': r'Self-Reflection - Patterns:\s*(.*?)(?=Self-Reflection - Strategy:|Reason:|$)',
                'reflection_strategy': r'Self-Reflection - Strategy:\s*(.*?)(?=Reason:|$)'
            }
            """
        else:
            # Original T3A - just extract reason
            reason_match = re.search(r'Reason:\s*(.*?)(?=Action:|$)', action_output, re.DOTALL)
            if reason_match:
                parsed['reason'] = reason_match.group(1).strip()
                reasoning_trace = parsed['reason']
            return reasoning_trace, parsed
        
        # Parse enhanced reasoning components
        for key, pattern in patterns.items():
            match = re.search(pattern, action_output, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                parsed[key] = content
                reasoning_trace += f"{key.replace('_', ' ').title()}: {content}\n"
        
        # Also extract the final reason
        reason_match = re.search(r'Reason:\s*(.*?)(?=Action:|$)', action_output, re.DOTALL)
        if reason_match:
            parsed['final_reason'] = reason_match.group(1).strip()
        
        return reasoning_trace.strip(), parsed

    def _generate_enhanced_summary(
        self,
        goal: str,
        action: str,
        reason: str,
        before_elements: str,
        after_elements: str,
        parsed_reasoning: dict
    ) -> str:
        """Generate enhanced summary incorporating reasoning traces."""
        
        # Start with original summary
        if self.enable_summarization:
            summary_prompt = _summarize_prompt(goal, action, reason, before_elements, after_elements)
            summary, is_safe, raw_response = self.llm.predict(summary_prompt)
            
            if is_safe == False:
                summary = "Summary triggered LLM safety classifier."
            elif not raw_response:
                summary = 'Error calling LLM in summarization phase.'
        else:
            summary = f'Action: {action}. Reason: {reason}'
        
        # Enhance with reasoning insights
        if parsed_reasoning:
            reasoning_insights = []
            
            if self.prompt_type == "chain_of_thought":
                if 'situation_analysis' in parsed_reasoning:
                    reasoning_insights.append(f"Situation: {parsed_reasoning['situation_analysis'][:100]}...")
                if 'risk_evaluation' in parsed_reasoning:
                    reasoning_insights.append(f"Risk considered: {parsed_reasoning['risk_evaluation'][:100]}...")
                    
            elif self.prompt_type == "self_reflection":
                if 'progress_reflection' in parsed_reasoning:
                    reasoning_insights.append(f"Progress: {parsed_reasoning['progress_reflection'][:100]}...")
                if 'error_analysis' in parsed_reasoning:
                    reasoning_insights.append(f"Error insight: {parsed_reasoning['error_analysis'][:100]}...")
            
            if reasoning_insights:
                summary += f" Enhanced reasoning: {'; '.join(reasoning_insights)}"
        
        return summary

    def get_performance_metrics(self) -> dict:
        """Get enhanced performance metrics."""
        total_steps = len(self.action_success_history)
        successful_steps = sum(self.action_success_history)
        
        return {
            'prompt_type': self.prompt_type,
            'total_steps': total_steps,
            'successful_steps': successful_steps,
            'step_success_rate': successful_steps / total_steps if total_steps > 0 else 0,
            'error_counts': dict(self.error_counts),
            'reasoning_quality': self._assess_reasoning_quality(),
            'history_length': len(self.history)
        }
    
    def _assess_reasoning_quality(self) -> dict:
        """Assess the quality of reasoning traces."""
        if not self.reasoning_traces:
            return {'quality_score': 0, 'completeness': 0}
        
        # Simple heuristic for reasoning quality
        total_components = 0
        filled_components = 0
        
        for trace in self.reasoning_traces:
            if self.prompt_type == "chain_of_thought":
                expected_keys = ['situation_analysis', 'goal_assessment', 'action_planning', 'risk_evaluation']
            elif self.prompt_type == "self_reflection":
                expected_keys = ['progress_reflection', 'error_analysis', 'strategy_assessment', 'pattern_recognition']
            elif self.prompt_type == "combined":
                #expected_keys = ['cot_situation', 'cot_goal', 'reflection_progress', 'reflection_strategy']
                expected_keys = ['situation', 'goal_status', 'options', 'reflection', 'decision']
            else:
                expected_keys = ['reason']
            
            total_components += len(expected_keys)
            filled_components += sum(1 for key in expected_keys if key in trace and len(trace[key]) > 10)
        
        completeness = filled_components / total_components if total_components > 0 else 0
        
        # Quality score based on completeness and length
        avg_length = sum(len(str(trace)) for trace in self.reasoning_traces) / len(self.reasoning_traces)
        quality_score = min(1.0, (completeness * 0.7) + (min(avg_length / 500, 1.0) * 0.3))
        
        return {
            'quality_score': quality_score,
            'completeness': completeness,
            'avg_reasoning_length': avg_length
        }
    
    def _build_backtrack_context(self) -> str:
        """Build context about potential need for backtracking"""
        context = ""
        
        # Check for repeated actions
        if len(self.recent_actions) >= self.backtrack_threshold:
            recent_action_types = [action.get('action_type', '') for action in self.recent_actions[-self.backtrack_threshold:]]
            recent_indices = [action.get('index', -1) for action in self.recent_actions[-self.backtrack_threshold:] if 'index' in action]
            
            # Check for repeated action types
            if len(set(recent_action_types)) <= 2:
                context += f"⚠️ BACKTRACK ALERT: You have repeated similar actions ({', '.join(recent_action_types[-3:])}). Consider using navigate_back or trying a different approach.\n"
            
            # Check for repeated clicks on same element
            if len(recent_indices) >= 2 and len(set(recent_indices)) == 1:
                context += f"⚠️ STUCK ALERT: You have clicked the same element (index {recent_indices[-1]}) multiple times. This suggests you may be stuck. Try navigate_back.\n"
        
        # Check stuck counter
        if self.stuck_counter >= 2:
            context += f"⚠️ EXECUTION FAILURES: You have had {self.stuck_counter} consecutive failed actions. Consider changing your strategy or using navigate_back.\n"
        
        return context
    
    def _build_completion_context(self, goal: str) -> str:
        """Build context about potential goal completion"""
        context = ""
        
        # Check for completion indicators
        if self.completion_indicators:
            recent_indicators = self.completion_indicators[-3:]  # Last 3 potential indicators
            if len(recent_indicators) >= 2:
                context += f"🎯 COMPLETION CHECK: Recent screens may indicate goal completion. Carefully check if '{goal}' is already achieved.\n"
        
        # Check for goal-specific completion patterns
        goal_lower = goal.lower()
        
        if 'turn on' in goal_lower or 'enable' in goal_lower:
            context += "🎯 ENABLE TASK: Look for toggle switches that are already 'ON' or enabled states. If you see the feature is already enabled, use status 'complete'.\n"
        
        if 'uninstall' in goal_lower:
            context += "🎯 UNINSTALL TASK: If the app is no longer visible in the app list, it may already be uninstalled. Check carefully before continuing.\n"
        
        if 'open' in goal_lower or 'launch' in goal_lower:
            context += "🎯 OPEN TASK: If you can see the app is already open (check the current screen), use status 'complete'.\n"
        
        return context
    
    def _track_action_for_backtracking(self, action):
        """Track action for backtracking detection"""
        action_info = {
            'action_type': action.action_type,
            'step': self.step_count
        }
        
        # Add index for actions that have it
        if hasattr(action, 'index') and action.index is not None:
            action_info['index'] = action.index
        
        # Add text for input actions
        if hasattr(action, 'text') and action.text:
            action_info['text'] = action.text
        
        self.recent_actions.append(action_info)
        
        # Keep only recent actions for analysis
        if len(self.recent_actions) > 10:
            self.recent_actions = self.recent_actions[-10:]
    
    # Additional helper method to safely extract text from UIElement
    def _extract_ui_element_text(self, element) -> str:
        """Safely extract text content from a UIElement object"""
        text_parts = []
    
        # Common text attributes in UIElement
        text_attributes = ['text', 'content_desc', 'resource_id', 'class_name']
    
        for attr in text_attributes:
            if hasattr(element, attr):
                value = getattr(element, attr)
                if value and isinstance(value, str) and value.strip():
                    text_parts.append(value.strip())
    
        return ' '.join(text_parts)

    # Updated completion check method using the helper
    def _check_completion_indicators(self, goal: str, ui_elements):
        """Improved version with better text extraction"""
        goal_lower = goal.lower()
        completion_signals = []
    
        for element in ui_elements:
            element_text = self._extract_ui_element_text(element).lower()
        
            if not element_text:
                continue
            
            # Check for "ON" states for enable/turn on tasks
            if ('turn on' in goal_lower or 'enable' in goal_lower):
                on_indicators = ['on', 'enabled', 'active', 'connected', 'wifi_on', 'toggle_on']
                if any(indicator in element_text for indicator in on_indicators):
                    completion_signals.append(f"Found 'enabled' indicator: {element_text[:50]}")
        
            # Check for "OFF" states for disable/turn off tasks  
            if ('turn off' in goal_lower or 'disable' in goal_lower):
                off_indicators = ['off', 'disabled', 'inactive', 'disconnected', 'wifi_off', 'toggle_off']
                if any(indicator in element_text for indicator in off_indicators):
                    completion_signals.append(f"Found 'disabled' indicator: {element_text[:50]}")
        
            # Check for uninstall completion
            if 'uninstall' in goal_lower:
                uninstall_indicators = ['uninstalled', 'removed', 'not installed', 'install'] # 'install' suggests app not present
                if any(indicator in element_text for indicator in uninstall_indicators):
                    completion_signals.append(f"Found uninstall indicator: {element_text[:50]}")
        
            # Check for general completion confirmations
            completion_words = ['done', 'complete', 'completed', 'finished', 'success', 'successful']
            if any(word in element_text for word in completion_words):
                completion_signals.append(f"Found completion indicator: {element_text[:50]}")
    
        if completion_signals:
            self.completion_indicators.extend(completion_signals)
        
            # Keep only recent indicators
            if len(self.completion_indicators) > 20:
                self.completion_indicators = self.completion_indicators[-20:]