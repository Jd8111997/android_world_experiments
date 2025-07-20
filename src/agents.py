#!/usr/bin/env python3
"""
Enhanced Android Agent with LLM Integration
"""

import re
import json
import logging
from typing import Dict, Any, List
from collections import defaultdict

from android_world.agents import base_agent
from android_world.env import json_action
from framework import generate_ui_elements_description_list_full
from utils import parse_ui_elements_from_description
from llm_client import LLMClient
from prompts import PromptTemplate

class EnhancedAndroidAgent(base_agent.EnvironmentInteractingAgent):
    """Enhanced Android agent with error handling and evaluation features"""
    
    def __init__(self, env, llm_client: LLMClient, prompt_template: PromptTemplate, 
                 template_name: str = "basic", name: str = "EnhancedAgent"):
        super().__init__(env, name=name, transition_pause=1.0)
        self.llm = llm_client
        self.prompt_template = prompt_template
        self.template_name = template_name
        self.step_count = 0
        self.episode_history = []
        self.error_counts = defaultdict(int)
        self.action_history = []
    
    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """Take one step toward the goal with enhanced error handling"""
        self.step_count += 1
        
        step_data = {
            'step_number': self.step_count,
            'goal': goal,
            'prompt': None,
            'llm_response': None,
            'action_attempted': None,
            'action_success': False,
            'reasoning': None,
            'error': None,
            'error_type': None,
            'ui_elements_count': 0,
            'clickable_elements_count': 0
        }
        
        try:
            # Get current state using proper base class method
            state = self.get_post_transition_state()
            ui_elements = self._extract_ui_elements_from_state(state)
            app_name = self._get_current_app_name()
            
            step_data['ui_elements_count'] = len(ui_elements)
            step_data['clickable_elements_count'] = len([e for e in ui_elements if e.get('clickable', False)])
            
            # Generate prompt with action history for reflection templates
            prompt = self.prompt_template.generate_prompt(
                template_name=self.template_name,
                goal=goal,
                ui_elements=ui_elements,
                step_num=self.step_count,
                app_name=app_name,
                action_history=self.action_history
            )
            step_data['prompt'] = prompt
            
            # Get LLM response
            llm_response = self.llm.generate_response(prompt, max_tokens=300)
            step_data['llm_response'] = llm_response
            
            # Parse and execute action
            action_result = self._parse_and_execute_action(llm_response, ui_elements)
            
            # Update step data
            step_data.update({
                'action_attempted': action_result.get('formatted_action', 'Unknown'),
                'action_success': action_result.get('success', False),
                'reasoning': action_result.get('reasoning', ''),
                'error': action_result.get('error', None),
                'error_type': action_result.get('error_type', None)
            })
            
            # Track error types
            if action_result.get('error_type'):
                self.error_counts[action_result['error_type']] += 1
            
            # Add to action history
            self.action_history.append({
                'step': self.step_count,
                'action': action_result.get('formatted_action', 'Unknown'),
                'success': action_result.get('success', False)
            })
            
            # Print step info
            print(f"Step {self.step_count}: {action_result.get('formatted_action', 'Unknown')}")
            if action_result.get('reasoning'):
                print(f"  Reasoning: {action_result['reasoning']}")
            if action_result.get('error'):
                print(f"  Error: {action_result['error']}")
            
            self.episode_history.append(step_data)
            
            return base_agent.AgentInteractionResult(
                done=action_result.get('done', False),
                data=step_data
            )
            
        except Exception as e:
            logging.error(f"Error in agent step {self.step_count}: {e}")
            step_data.update({
                'error': str(e),
                'error_type': 'execution_error',
                'action_success': False
            })
            self.error_counts['execution_error'] += 1
            self.episode_history.append(step_data)
            
            return base_agent.AgentInteractionResult(
                done=False,
                data=step_data
            )
    
    def _extract_ui_elements_from_state(self, state) -> List[Dict]:
        """Extract UI elements using Android World's T3A method"""
        try:
            ui_elements = state.ui_elements if hasattr(state, 'ui_elements') else []
            logical_screen_size = self.env.logical_screen_size
            
            # Generate description using T3A's method
            ui_description = generate_ui_elements_description_list_full(
                ui_elements, logical_screen_size
            )
            
            # Parse the description into our format
            parsed_elements = parse_ui_elements_from_description(ui_description)
            
            return parsed_elements
            
        except Exception as e:
            logging.warning(f"Failed to extract UI elements: {e}")
            return []
    
    def _get_current_app_name(self) -> str:
        """Get current app name"""
        try:
            if hasattr(self.env, 'get_current_activity'):
                activity = self.env.get_current_activity()
                if activity and '/' in activity:
                    package_name = activity.split('/')[0]
                    if '.' in package_name:
                        return package_name.split('.')[-1].title()
                    return package_name
                return activity or "Unknown"
            return "Unknown"
        except Exception as e:
            logging.debug(f"Could not get app name: {e}")
            return "Unknown"
    
    def _parse_and_execute_action(self, llm_response: str, ui_elements: List[Dict]) -> Dict[str, Any]:
        """Parse LLM response and execute the corresponding action with JSON format support"""
        
        response = llm_response.strip() if llm_response else ""
        
        # Check if LLM response is an error
        if response.startswith("ERROR:"):
            return {
                'action_type': 'error',
                'success': False,
                'reasoning': 'LLM API call failed',
                'error': response,
                'error_type': 'llm_api_error',
                'formatted_action': 'ERROR'
            }
        
        # Extract reasoning if present
        reasoning = self._extract_reasoning(response)
        
        # Try to extract JSON action from response
        json_action_match = re.search(r'\{[^}]*"action_type"[^}]*\}', response)
        if json_action_match:
            try:
                action_json = json.loads(json_action_match.group(0))
                return self._execute_json_action(action_json, ui_elements, reasoning)
            except json.JSONDecodeError:
                pass  # Fall through to legacy parsing
        
        # Legacy action parsing for backwards compatibility
        if 'CLICK(' in response.upper():
            return self._execute_click_action_legacy(response, ui_elements, reasoning)
        elif 'INPUT_TEXT(' in response.upper() or 'TYPE(' in response.upper():
            return self._execute_type_action_legacy(response, reasoning)
        elif 'SCROLL(' in response.upper():
            return self._execute_scroll_action_legacy(response, reasoning)
        elif 'NAVIGATE_BACK(' in response.upper():
            return self._execute_navigate_back_action(reasoning)
        elif 'NAVIGATE_HOME(' in response.upper():
            return self._execute_navigate_home_action(reasoning)
        elif 'STATUS(' in response.upper() or any(word in response.lower() for word in ['complete', 'done', 'finished']):
            return self._execute_status_action_legacy(response, reasoning)
        else:
            # Try to extract any quoted text as click target
            quoted_match = re.search(r'"([^"]*)"', response)
            if quoted_match:
                return self._click_by_text(quoted_match.group(1), ui_elements, reasoning)
        
        return {
            'action_type': 'unknown',
            'success': False,
            'reasoning': reasoning or response,
            'error': f'Could not parse action from: {response}',
            'error_type': 'action_parsing_error',
            'formatted_action': 'UNKNOWN_ACTION'
        }
    
    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from response for reflection templates"""
        reasoning_patterns = [
            r'Reason:\s*(.*?)(?=Action:|$)',
            r'REASONING:\s*(.*?)(?=ACTION:|$)',
            r'REFLECTION:\s*(.*?)(?=ACTION:|$)',
            r'EXPLANATION:\s*(.*?)(?=$)'
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _execute_json_action(self, action_json: Dict, ui_elements: List[Dict], reasoning: str) -> Dict[str, Any]:
        """Execute action from JSON format (T3A style)"""
        
        action_type = action_json.get('action_type')
        
        if action_type == 'click':
            return self._execute_json_click(action_json, ui_elements, reasoning)
        elif action_type == 'input_text':
            return self._execute_json_input_text(action_json, ui_elements, reasoning)
        elif action_type == 'long_press':
            return self._execute_json_long_press(action_json, ui_elements, reasoning)
        elif action_type == 'scroll':
            return self._execute_json_scroll(action_json, reasoning)
        elif action_type == 'navigate_back':
            return self._execute_navigate_back_action(reasoning)
        elif action_type == 'navigate_home':
            return self._execute_navigate_home_action(reasoning)
        elif action_type == 'open_app':
            return self._execute_json_open_app(action_json, reasoning)
        elif action_type == 'wait':
            return self._execute_wait_action(reasoning)
        elif action_type == 'status':
            return self._execute_json_status(action_json, reasoning)
        elif action_type == 'answer':
            return self._execute_json_answer(action_json, reasoning)
        else:
            return {
                'action_type': action_type,
                'success': False,
                'reasoning': reasoning,
                'error': f'Unknown action type: {action_type}',
                'error_type': 'action_parsing_error',
                'formatted_action': f'UNKNOWN({action_type})'
            }
    
    def _execute_json_click(self, action_json: Dict, ui_elements: List[Dict], reasoning: str) -> Dict[str, Any]:
        """Execute JSON click action with proper index validation"""
        
        index = action_json.get('index')
        if index is None:
            return {
                'action_type': 'click',
                'success': False,
                'reasoning': reasoning,
                'error': 'Click action missing index parameter',
                'error_type': 'action_parsing_error',
                'formatted_action': 'CLICK(missing_index)'
            }
        
        # Find element with matching index
        target_element = None
        for element in ui_elements:
            if element.get('index') == index:
                target_element = element
                break
        
        if not target_element:
            available_indices = [elem.get('index') for elem in ui_elements if elem.get('clickable', False)]
            return {
                'action_type': 'click',
                'success': False,
                'reasoning': reasoning,
                'error': f'No element found with index {index}. Available: {available_indices}',
                'error_type': 'element_not_found',
                'formatted_action': f'CLICK(index={index})'
            }
        
        # Check if element is clickable
        if not target_element.get('clickable', False):
            return {
                'action_type': 'click',
                'success': False,
                'reasoning': reasoning,
                'error': f'Element at index {index} is not clickable',
                'error_type': 'element_not_clickable',
                'formatted_action': f'CLICK(index={index})'
            }
        
        # Execute the click
        try:
            action = json_action.JSONAction(action_type="click", index=index)
            self.env.execute_action(action)
            
            element_text = target_element.get('text', '') or target_element.get('content_desc', '')
            return {
                'action_type': 'click',
                'success': True,
                'reasoning': reasoning or f'Successfully clicked on "{element_text}" (index {index})',
                'element_index': index,
                'formatted_action': f'CLICK(index={index})'
            }
        except Exception as e:
            return {
                'action_type': 'click',
                'success': False,
                'reasoning': reasoning,
                'error': f'Click execution failed: {str(e)}',
                'error_type': 'execution_error',
                'formatted_action': f'CLICK(index={index})'
            }
    
    # Additional JSON action methods would go here...
    # (Due to length constraints, I'm showing the pattern for click)
    
    # Legacy action methods for backwards compatibility
    def _execute_click_action_legacy(self, response: str, ui_elements: List[Dict], reasoning: str) -> Dict[str, Any]:
        """Legacy click action for backwards compatibility"""
        
        click_match = re.search(r'CLICK\s*\(\s*["\']([^"\']*)["\']', response, re.IGNORECASE)
        if not click_match:
            return {
                'action_type': 'click',
                'success': False,
                'reasoning': reasoning,
                'error': 'Could not parse click target',
                'error_type': 'action_parsing_error',
                'formatted_action': 'CLICK(?)'
            }
        
        target_text = click_match.group(1)
        return self._click_by_text(target_text, ui_elements, reasoning)
    
    def _click_by_text(self, target_text: str, ui_elements: List[Dict], reasoning: str) -> Dict[str, Any]:
        """Find and click element by text (legacy support)"""
        
        # Try exact match on clickable elements
        for element in ui_elements:
            if not element.get('clickable', False):
                continue
                
            element_text = element.get('text', '').strip()
            content_desc = element.get('content_desc', '').strip()
            index = element.get('index')
            
            if (element_text and element_text.lower() == target_text.lower()) or \
               (content_desc and content_desc.lower() == target_text.lower()):
                
                try:
                    action = json_action.JSONAction(action_type="click", index=index)
                    self.env.execute_action(action)
                    
                    return {
                        'action_type': 'click',
                        'success': True,
                        'reasoning': reasoning or f'Clicked on "{element_text or content_desc}" (index {index})',
                        'element_index': index,
                        'formatted_action': f'CLICK("{target_text}")'
                    }
                except Exception as e:
                    return {
                        'action_type': 'click',
                        'success': False,
                        'reasoning': reasoning,
                        'error': f'Click execution failed: {str(e)}',
                        'error_type': 'execution_error',
                        'formatted_action': f'CLICK("{target_text}")'
                    }
        
        # Show available elements if no match found
        clickable_elements = [elem for elem in ui_elements if elem.get('clickable', False)]
        available = [elem.get('text') or elem.get('content_desc') for elem in clickable_elements 
                    if elem.get('text') or elem.get('content_desc')]
        
        return {
            'action_type': 'click',
            'success': False,
            'reasoning': reasoning,
            'error': f'Element "{target_text}" not found. Available: {available[:5]}',
            'error_type': 'element_not_found',
            'formatted_action': f'CLICK("{target_text}")'
        }
    
    # Additional implementation methods...
    def _execute_navigate_back_action(self, reasoning: str) -> Dict[str, Any]:
        """Execute a NAVIGATE_BACK action"""
        try:
            action = json_action.JSONAction(action_type="navigate_back")
            self.env.execute_action(action)
            
            return {
                'action_type': 'navigate_back',
                'success': True,
                'reasoning': reasoning or 'Navigated back',
                'formatted_action': 'NAVIGATE_BACK()'
            }
        except Exception as e:
            return {
                'action_type': 'navigate_back',
                'success': False,
                'reasoning': reasoning,
                'error': f'Navigate back execution failed: {str(e)}',
                'error_type': 'execution_error',
                'formatted_action': 'NAVIGATE_BACK()'
            }
    
    def _execute_navigate_home_action(self, reasoning: str) -> Dict[str, Any]:
        """Execute a NAVIGATE_HOME action"""
        try:
            action = json_action.JSONAction(action_type="navigate_home")
            self.env.execute_action(action)
            
            return {
                'action_type': 'navigate_home',
                'success': True,
                'reasoning': reasoning or 'Navigated to home',
                'formatted_action': 'NAVIGATE_HOME()'
            }
        except Exception as e:
            return {
                'action_type': 'navigate_home',
                'success': False,
                'reasoning': reasoning,
                'error': f'Navigate home execution failed: {str(e)}',
                'error_type': 'execution_error',
                'formatted_action': 'NAVIGATE_HOME()'
            }
    
    def _execute_wait_action(self, reasoning: str) -> Dict[str, Any]:
        """Execute a WAIT action"""
        try:
            action = json_action.JSONAction(action_type="wait")
            self.env.execute_action(action)
            
            return {
                'action_type': 'wait',
                'success': True,
                'reasoning': reasoning or 'Waited for UI transition',
                'formatted_action': 'WAIT()'
            }
        except Exception as e:
            return {
                'action_type': 'wait',
                'success': False,
                'reasoning': reasoning,
                'error': f'Wait execution failed: {str(e)}',
                'error_type': 'execution_error',
                'formatted_action': 'WAIT()'
            }