#!/usr/bin/env python3
"""
Enhanced prompt template system with T3A-inspired design
"""

import random
from typing import List, Dict

class PromptTemplate:
    """Enhanced prompt templates with few-shot examples and self-reflection"""
    
    def __init__(self):
        self.templates = {
            "basic": self._basic_template,
            "detailed": self._detailed_template,
            "few_shot": self._few_shot_template,
            "chain_of_thought": self._chain_of_thought_template,
            "self_reflection": self._self_reflection_template,
            "few_shot_reflection": self._few_shot_reflection_template
        }
        
        # Load few-shot examples from prompts directory
        self.few_shot_examples = self._load_few_shot_examples()
    
    def _load_few_shot_examples(self) -> List[Dict]:
        """Load few-shot examples (in practice, these would come from prompts/ directory)"""
        return [
            {
                "goal": "Uninstall the Slack app",
                "ui_elements": [
                    {"index": 0, "text": "Settings", "clickable": True},
                    {"index": 1, "text": "Apps", "clickable": True},
                    {"index": 2, "text": "Battery", "clickable": True}
                ],
                "action": '{"action_type": "click", "index": 1}',
                "reasoning": "To uninstall an app, I need to access the Apps section. I can see 'Apps' at index 1, so I'll click on it."
            },
            {
                "goal": "Turn on Wi-Fi",
                "ui_elements": [
                    {"index": 0, "text": "Wi-Fi", "clickable": True},
                    {"index": 1, "text": "Bluetooth", "clickable": True},
                    {"index": 2, "text": "Mobile data", "clickable": True}
                ],
                "action": '{"action_type": "click", "index": 0}',
                "reasoning": "I can see 'Wi-Fi' option at index 0. Clicking on it should take me to Wi-Fi settings where I can turn it on."
            },
            {
                "goal": "Add a new contact",
                "ui_elements": [
                    {"index": 0, "text": "Search contacts", "clickable": True},
                    {"index": 1, "text": "+", "clickable": True},
                    {"index": 2, "text": "Recent", "clickable": True}
                ],
                "action": '{"action_type": "click", "index": 1}',
                "reasoning": "To add a new contact, I should click the '+' button at index 1 which is the standard way to add new items."
            },
            {
                "goal": "Type 'Hello World' in the search box",
                "ui_elements": [
                    {"index": 0, "text": "Search", "clickable": True},
                    {"index": 1, "text": "Menu", "clickable": True}
                ],
                "action": '{"action_type": "input_text", "text": "Hello World", "index": 0}',
                "reasoning": "I need to type 'Hello World' in the search box. The search element is at index 0, so I'll use input_text action."
            }
        ]
    
    def generate_prompt(self, template_name: str, goal: str, ui_elements: List[Dict], 
                       step_num: int, **kwargs) -> str:
        """Generate prompt using specified template"""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(self.templates.keys())}")
        return self.templates[template_name](goal, ui_elements, step_num, **kwargs)
    
    def _build_ui_elements_description(self, ui_elements: List[Dict]) -> str:
        """Build properly formatted UI elements description"""
        if not ui_elements:
            return "No UI elements available"
        
        description = ""
        for element in ui_elements:
            index = element.get('index', 'N/A')
            text = element.get('text', '').strip()
            content_desc = element.get('content_desc', '').strip()
            clickable = element.get('clickable', False)
            
            # Build element description similar to T3A format
            element_info = f"UI element {index}: "
            
            # Add text information
            if text:
                element_info += f"text='{text}', "
            elif content_desc:
                element_info += f"content_description='{content_desc}', "
            else:
                element_info += "text='', "
            
            # Add interaction information
            element_info += f"clickable={clickable}"
            
            # Only include clickable elements or important non-clickable ones
            if clickable or text or content_desc:
                description += element_info + "\n"
        
        return description.strip() if description else "No interactive elements found"
    
    def _basic_template(self, goal: str, ui_elements: List[Dict], step_num: int, **kwargs) -> str:
        """Enhanced basic template following T3A structure"""
        
        ui_description = self._build_ui_elements_description(ui_elements)
        
        return f"""You are an agent who can operate an Android phone on behalf of a user. Based on user's goal/request, you will complete tasks step by step by performing actions on the phone.

Current Goal: {goal}

Available UI Elements:
{ui_description}

Available Actions (output in JSON format):
- Click on a UI element: {{"action_type": "click", "index": <target_index>}}
- Type text into an element: {{"action_type": "input_text", "text": "<text>", "index": <target_index>}}
- Scroll the screen: {{"action_type": "scroll", "direction": "<up|down|left|right>"}}
- Navigate back: {{"action_type": "navigate_back"}}
- Navigate home: {{"action_type": "navigate_home"}}
- Open an app: {{"action_type": "open_app", "app_name": "<name>"}}
- Wait for screen update: {{"action_type": "wait"}}
- Complete task: {{"action_type": "status", "goal_status": "complete"}}
- Mark as infeasible: {{"action_type": "status", "goal_status": "infeasible"}}

Important Guidelines:
- The index parameter must be from the UI elements list above
- Use exact indices shown in the list
- Choose the action that best progresses toward the goal

Your response format:
Reason: [Brief explanation of why you chose this action]
Action: {{"action_type": "...", ...}}

Your Answer:"""
    
    def _detailed_template(self, goal: str, ui_elements: List[Dict], step_num: int, **kwargs) -> str:
        """Enhanced detailed template with comprehensive guidance"""
        
        app_name = kwargs.get('app_name', 'Unknown')
        ui_description = self._build_ui_elements_description(ui_elements)
        
        # Build action history summary
        history = kwargs.get('action_history', [])
        history_text = ""
        if history:
            history_text = "Previous Actions:\n"
            for i, action in enumerate(history[-3:], 1):
                action_desc = action.get('action', 'Unknown action')
                success = "✓" if action.get('success', False) else "✗"
                history_text += f"  Step {len(history) - 3 + i}: {action_desc} {success}\n"
            history_text += "\n"
        
        return f"""You are an autonomous Android agent. Your task is to complete user requests by performing actions step by step.

CURRENT CONTEXT:
Goal: {goal}
Current App: {app_name}
Step Number: {step_num}

{history_text}CURRENT SCREEN STATE:
{ui_description}

ACTION GUIDELINES:
- Use open_app action to launch apps (preferred over navigating through app drawer)
- For text input, use input_text action instead of clicking keyboard keys
- Index parameter must match exactly with UI elements above
- Consider scrolling to reveal more content if needed
- If goal is already achieved, use status action with "complete"
- If task seems impossible, use status action with "infeasible"

AVAILABLE ACTIONS:
1. Click: {{"action_type": "click", "index": <number>}}
2. Input Text: {{"action_type": "input_text", "text": "<text>", "index": <number>}}
3. Long Press: {{"action_type": "long_press", "index": <number>}}
4. Scroll: {{"action_type": "scroll", "direction": "<up|down|left|right>"}}
5. Navigate Back: {{"action_type": "navigate_back"}}
6. Navigate Home: {{"action_type": "navigate_home"}}
7. Open App: {{"action_type": "open_app", "app_name": "<name>"}}
8. Wait: {{"action_type": "wait"}}
9. Complete: {{"action_type": "status", "goal_status": "complete"}}
10. Infeasible: {{"action_type": "status", "goal_status": "infeasible"}}

OUTPUT FORMAT:
Reason: [Detailed explanation of your reasoning]
Action: [JSON action from the list above]

Your Answer:"""
    
    def _few_shot_template(self, goal: str, ui_elements: List[Dict], step_num: int, **kwargs) -> str:
        """Enhanced few-shot template with proper examples"""
        
        # Select relevant examples
        selected_examples = self.few_shot_examples[:3]
        
        examples_text = "EXAMPLES OF GOOD AGENT BEHAVIOR:\n\n"
        for i, example in enumerate(selected_examples, 1):
            # Format the example UI elements
            example_ui = ""
            for elem in example['ui_elements']:
                if elem.get('clickable', True):
                    example_ui += f"UI element {elem['index']}: text='{elem['text']}', clickable=True\n"
            
            examples_text += f"Example {i}:\n"
            examples_text += f"Goal: {example['goal']}\n"
            examples_text += f"UI Elements:\n{example_ui}\n"
            examples_text += f"Reason: {example['reasoning']}\n"
            examples_text += f"Action: {example['action']}\n\n"
        
        # Current situation
        ui_description = self._build_ui_elements_description(ui_elements)
        
        current_text = f"""NOW SOLVE THIS TASK:

Goal: {goal}
Step: {step_num}

Current UI Elements:
{ui_description}

Follow the same format as the examples above. Choose the best action to progress toward the goal.

Your response format:
Reason: [Why you chose this action]
Action: [JSON action]

Your Answer:"""
        
        return examples_text + current_text
    
    def _chain_of_thought_template(self, goal: str, ui_elements: List[Dict], step_num: int, **kwargs) -> str:
        """Enhanced chain of thought template with structured reasoning"""
        
        ui_description = self._build_ui_elements_description(ui_elements)
        app_name = kwargs.get('app_name', 'Unknown')
        
        return f"""You are an Android automation agent. Think through this task step by step using structured reasoning.

TASK ANALYSIS:
Goal: {goal}
Current App: {app_name}
Step: {step_num}

CURRENT SCREEN:
{ui_description}

STRUCTURED REASONING PROCESS:
Please think through this systematically:

1. SITUATION ANALYSIS:
   - What do I see on the current screen?
   - What app am I currently in?
   - What UI elements are available to interact with?

2. GOAL ASSESSMENT:
   - What is my ultimate objective?
   - What is the next logical step toward this goal?
   - Are there any intermediate steps needed?

3. ACTION PLANNING:
   - Which UI element should I interact with?
   - What type of action is most appropriate?
   - What are the expected outcomes?

4. RISK EVALUATION:
   - Could this action fail or cause problems?
   - Are there alternative approaches if this doesn't work?
   - Is the element I want to click actually visible and clickable?

AVAILABLE ACTIONS:
- {{"action_type": "click", "index": <number>}}
- {{"action_type": "input_text", "text": "<text>", "index": <number>}}
- {{"action_type": "scroll", "direction": "<direction>"}}
- {{"action_type": "navigate_back"}}
- {{"action_type": "navigate_home"}}
- {{"action_type": "open_app", "app_name": "<name>"}}
- {{"action_type": "wait"}}
- {{"action_type": "status", "goal_status": "<complete|infeasible>"}}

RESPONSE FORMAT:
Situation Analysis: [Your analysis of current situation]
Goal Assessment: [How this relates to your objective]
Action Planning: [What you plan to do and why]
Risk Evaluation: [Potential issues and alternatives]
Final Decision: [Your chosen action with reasoning]
Action: [JSON format action]

Your Answer:"""
    
    def _self_reflection_template(self, goal: str, ui_elements: List[Dict], step_num: int, **kwargs) -> str:
        """Enhanced self-reflection template with progress tracking"""
        
        history = kwargs.get('action_history', [])
        ui_description = self._build_ui_elements_description(ui_elements)
        
        # Build detailed history analysis
        history_analysis = ""
        if history:
            recent_actions = history[-3:]
            history_analysis = "RECENT ACTION HISTORY:\n"
            for i, action in enumerate(recent_actions):
                step_num_hist = len(history) - len(recent_actions) + i + 1
                action_desc = action.get('action', 'Unknown')
                success = action.get('success', False)
                result = "SUCCESS" if success else "FAILED"
                history_analysis += f"  Step {step_num_hist}: {action_desc} -> {result}\n"
            history_analysis += "\n"
        else:
            history_analysis = "RECENT ACTION HISTORY:\nThis is the first step - no previous actions.\n\n"
        
        return f"""You are a self-reflective Android agent. Before taking action, analyze your progress and learn from previous steps.

CURRENT TASK:
Goal: {goal}
Current Step: {step_num}

{history_analysis}CURRENT SCREEN STATE:
{ui_description}

SELF-REFLECTION PROCESS:
Please reflect on your progress before deciding on the next action:

1. PROGRESS EVALUATION:
   - How much progress have I made toward the goal?
   - What has worked well in previous steps?
   - What challenges or failures have I encountered?

2. PATTERN RECOGNITION:
   - Are there patterns in the UI that suggest the right path?
   - Have I seen similar screens or elements before?
   - What do I know about Android app navigation patterns?

3. STRATEGY ASSESSMENT:
   - Is my current approach working effectively?
   - Should I continue with the same strategy or try something different?
   - Are there alternative paths I should consider?

4. ERROR ANALYSIS:
   - What could have caused previous failures?
   - How can I avoid repeating mistakes?
   - Are there elements I might have overlooked?

5. NEXT STEP PLANNING:
   - Given my reflection, what is the best next action?
   - What are the potential risks and benefits?
   - How will this action move me closer to the goal?

AVAILABLE ACTIONS (use exact JSON format):
- {{"action_type": "click", "index": <number>}}
- {{"action_type": "input_text", "text": "<text>", "index": <number>}}
- {{"action_type": "long_press", "index": <number>}}
- {{"action_type": "scroll", "direction": "<up|down|left|right>"}}
- {{"action_type": "navigate_back"}}
- {{"action_type": "navigate_home"}}
- {{"action_type": "open_app", "app_name": "<name>"}}
- {{"action_type": "wait"}}
- {{"action_type": "status", "goal_status": "<complete|infeasible>"}}

RESPONSE FORMAT:
Progress Evaluation: [Your assessment of current progress]
Pattern Recognition: [What patterns you notice]
Strategy Assessment: [Whether your approach is working]
Error Analysis: [What you've learned from any failures]
Next Step Planning: [Your reasoning for the next action]
Action: [Your chosen action in JSON format]

Your Answer:"""
    
    def _few_shot_reflection_template(self, goal: str, ui_elements: List[Dict], step_num: int, **kwargs) -> str:
        """Combined few-shot and self-reflection template"""
        
        # Show one detailed example with reflection
        example_with_reflection = """EXAMPLE OF GOOD REFLECTIVE AGENT BEHAVIOR:

Goal: Enable Bluetooth
Previous Actions: Step 1: {"action_type": "click", "index": 0} -> SUCCESS (opened Settings)
Current UI Elements:
UI element 0: text='Connected devices', clickable=True
UI element 1: text='Apps', clickable=True
UI element 2: text='Battery', clickable=True

Progress Evaluation: I successfully opened Settings and can see 'Connected devices' option. This is good progress as Bluetooth settings are typically found there.
Pattern Recognition: Android typically organizes Bluetooth under 'Connected devices' or 'Connections'. The presence of this option confirms I'm on the right track.
Strategy Assessment: My approach of going through Settings is working well. I should continue with this path.
Error Analysis: No errors so far. The direct approach through Settings is proving effective.
Next Step Planning: I should click on 'Connected devices' as it's the most likely place to find Bluetooth settings.
Action: {"action_type": "click", "index": 0}

NOW SOLVE THIS TASK:
"""
        
        # Current situation with reflection prompts
        history = kwargs.get('action_history', [])
        ui_description = self._build_ui_elements_description(ui_elements)
        
        history_text = ""
        if history:
            recent_actions = [f'Step {i+1}: {action.get("action", "Unknown")} -> {"SUCCESS" if action.get("success", False) else "FAILED"}' 
                            for i, action in enumerate(history[-2:])]
            history_text = f"Previous Actions: {', '.join(recent_actions)}\n"
        
        current_text = f"""Goal: {goal}
{history_text}Current UI Elements:
{ui_description}

Follow the same reflective approach as the example above. Think through your progress and choose the best action.

Progress Evaluation: [Assess your current progress]
Pattern Recognition: [Identify UI/navigation patterns]
Strategy Assessment: [Evaluate your approach]
Error Analysis: [Learn from any previous issues]
Next Step Planning: [Plan your next move]
Action: [JSON format action]

Your Answer:"""
        
        return example_with_reflection + current_text