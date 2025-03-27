#!/usr/bin/env python3
import json
import os
import sys
from collections import Counter
from datetime import datetime

def analyze_json_structure(file_path):
    """Analyze the structure of conversations.json file"""
    print(f"Analyzing file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Check if it's a list
    if not isinstance(data, list):
        print(f"File does not contain a list. Type: {type(data)}")
        return
    
    print(f"Found {len(data)} items in the list")
    
    # Count occurrence of various fields
    field_presence = Counter()
    create_time_types = Counter()
    model_types = Counter()
    mapping_lengths = []
    empty_mappings = 0
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Item {i} is not a dictionary. Type: {type(item)}")
            continue
        
        # Check fields
        for field in item.keys():
            field_presence[field] += 1
        
        # Check create_time field type
        if 'create_time' in item:
            create_time = item['create_time']
            create_time_types[str(type(create_time))] += 1
            
            if isinstance(create_time, str) and 'T' in create_time:
                print(f"String timestamp example: {create_time}")
            elif isinstance(create_time, (int, float)):
                print(f"Numeric timestamp example: {create_time}")
        
        # Check model field types
        if 'model_slug' in item:
            model = item['model_slug']
            model_types[str(type(model))] += 1
            print(f"Model example: {model}")
        
        # Check mapping structure
        if 'mapping' in item:
            mapping = item['mapping']
            if not mapping:
                empty_mappings += 1
            else:
                mapping_lengths.append(len(mapping))
                
                # Look at the first node
                first_node_id = next(iter(mapping))
                first_node = mapping[first_node_id]
                
                if i == 0:  # Print only for the first item
                    print("\nExample node structure:")
                    print(json.dumps(first_node, indent=2))
                    
                    if 'message' in first_node and first_node['message'] is not None:
                        message = first_node['message']
                        if 'content' in message:
                            content = message['content']
                            print("\nContent structure type:", type(content))
                            if isinstance(content, dict) and 'parts' in content:
                                parts = content['parts']
                                print("Parts type:", type(parts))
                                if parts and len(parts) > 0:
                                    print("First part type:", type(parts[0]))
                                    print("First part value:", parts[0])
    
    # Print statistics
    print("\nField presence statistics:")
    total_items = len(data)
    for field, count in field_presence.most_common():
        print(f"  {field}: {count}/{total_items} ({count/total_items*100:.1f}%)")
    
    print("\nCreate time field types:")
    for type_name, count in create_time_types.most_common():
        print(f"  {type_name}: {count}")
    
    print("\nModel field types:")
    for type_name, count in model_types.most_common():
        print(f"  {type_name}: {count}")
    
    if mapping_lengths:
        print(f"\nMapping statistics:")
        print(f"  Empty mappings: {empty_mappings}/{total_items}")
        print(f"  Average mapping length: {sum(mapping_lengths)/len(mapping_lengths):.1f} nodes")
        print(f"  Min mapping length: {min(mapping_lengths)} nodes")
        print(f"  Max mapping length: {max(mapping_lengths)} nodes")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        file_path = "openai-conversations/conversations.json"
    else:
        file_path = sys.argv[1]
    
    analyze_json_structure(file_path)