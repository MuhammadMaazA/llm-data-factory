#!/usr/bin/env python3
"""Quick test script to generate a small amount of synthetic data."""

import os
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def main():
    """Generate a small test dataset."""
    print("Generating small test dataset...")
    
    # Set up OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Set up paths
    base_dir = Path(__file__).parent
    seed_file = base_dir / "data" / "seed_examples.json"
    output_file = base_dir / "data" / "test_synthetic_data.json"
    
    # Load seed examples
    with open(seed_file, 'r') as f:
        examples = json.load(f)
    
    # Create a simple prompt
    prompt = """Generate 5 realistic customer support tickets in JSON format. Return a JSON object with a "tickets" array.

Each ticket should have:
- ticket_id (TICKET-XXX format)  
- customer_message (realistic issue description)
- category (one of: Authentication, Technical, Billing, Feature Request, General)
- priority (one of: Low, Medium, High, Critical)
- customer_id (CUST-XXX format)

Example format:
{
  "tickets": [
    {
      "ticket_id": "TICKET-001",
      "customer_message": "I cannot log into my account",
      "category": "Authentication", 
      "priority": "High",
      "customer_id": "CUST-001"
    }
  ]
}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates realistic customer support tickets in JSON format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        
        if 'tickets' in data:
            tickets = data['tickets']
        else:
            tickets = [data] if isinstance(data, dict) else []
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(tickets, f, indent=2)
            
        print(f"Generated {len(tickets)} test tickets")
        print(f"Test dataset saved to: {output_file}")
        
        # Show first ticket as example
        if tickets:
            print("\nFirst generated ticket:")
            print(json.dumps(tickets[0], indent=2))
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
