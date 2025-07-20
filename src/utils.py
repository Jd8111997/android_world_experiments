#!/usr/bin/env python3
"""
Utility functions for Android World evaluation
"""

import re
import logging
from typing import List, Dict

def parse_ui_elements_from_description(ui_description: str) -> List[Dict]:
    """Parse the UI elements description string from T3A into our format"""
    ui_elements = []
    
    if not ui_description or ui_description == "Not available":
        return ui_elements
    
    lines = ui_description.strip().split('\n')
    
    for line in lines:
        if line.startswith('UI element '):
            try:
                # Extract index
                index_part = line.split(':')[0]
                index = int(index_part.replace('UI element ', ''))
                
                # Parse the UIElement string
                element_dict = {
                    'index': index,
                    'text': '',
                    'content_desc': '',
                    'clickable': False,
                    'class': '',
                    'resource_id': ''
                }
                
                # Extract text
                if 'text=' in line:
                    text_match = re.search(r"text='([^']*)'", line)
                    if text_match:
                        text_val = text_match.group(1)
                        if text_val and text_val != 'None':
                            element_dict['text'] = text_val
                
                # Extract content_description  
                if 'content_description=' in line:
                    desc_match = re.search(r"content_description='([^']*)'", line)
                    if desc_match:
                        desc_val = desc_match.group(1)
                        if desc_val and desc_val != 'None':
                            element_dict['content_desc'] = desc_val
                
                # Extract class_name
                if 'class_name=' in line:
                    class_match = re.search(r"class_name='([^']*)'", line)
                    if class_match:
                        element_dict['class'] = class_match.group(1)
                
                # Extract is_clickable
                if 'is_clickable=' in line:
                    clickable_match = re.search(r"is_clickable=(\w+)", line)
                    if clickable_match:
                        element_dict['clickable'] = clickable_match.group(1) == 'True'
                
                ui_elements.append(element_dict)
                
            except Exception as e:
                logging.debug(f"Failed to parse line: {line}, error: {e}")
                continue
    
    return ui_elements