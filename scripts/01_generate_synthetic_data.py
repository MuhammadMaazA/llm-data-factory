#!/usr/bin/env python3
"""
Synthetic Data Generation Script for LLM Data Factory

This script uses a powerful "Teacher" LLM (GPT-4) to generate synthetic customer support tickets
based on a small set of seed examples. The generated data will be used to fine-tune a smaller
"Student" model for classification tasks.
"""

import json
import logging
import os
import random
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from typing import Dict, List, Optional

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('synthetic_data_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TicketExample(BaseModel):
    """Pydantic model for ticket examples."""
    ticket_id: str
    customer_message: str
    category: str
    priority: str
    customer_id: str


class SyntheticDataGenerator:
    """Class to handle synthetic data generation using OpenAI API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the generator with API key."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.api_key
        self.client = openai.OpenAI()
        
        # Configuration
        self.categories = ["Urgent Bug", "Feature Request", "How-To Question"]
        self.priorities = ["High", "Medium", "Low"]
        self.target_samples = 1000
        self.batch_size = 10
        self.max_retries = 3
        self.retry_delay = 2
        
    def load_seed_examples(self, file_path: str) -> List[TicketExample]:
        """Load seed examples from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            examples = [TicketExample(**item) for item in data]
            logger.info(f"Loaded {len(examples)} seed examples from {file_path}")
            return examples
            
        except FileNotFoundError:
            logger.error(f"Seed examples file not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in seed examples file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading seed examples: {e}")
            raise
    
    def create_generation_prompt(self, examples: List[TicketExample]) -> str:
        """Create a comprehensive prompt for the Teacher model."""
        examples_text = "\n\n".join([
            f"Ticket ID: {ex.ticket_id}\n"
            f"Message: {ex.customer_message}\n"
            f"Category: {ex.category}\n"
            f"Priority: {ex.priority}\n"
            f"Customer ID: {ex.customer_id}"
            for ex in examples
        ])
        
        prompt = f"""You are an expert at generating realistic customer support tickets. Based on the following examples, generate {self.batch_size} new, diverse customer support tickets.

The tickets should be realistic, varied, and cover different scenarios. Each ticket should include:
- A realistic customer message describing their issue or request
- One of these categories: {', '.join(self.categories)}
- One of these priorities: {', '.join(self.priorities)}
- A unique ticket ID (format: TICKET-XXXXX)
- A unique customer ID (format: CUST-XXXXX)

Here are the example tickets to learn from:

{examples_text}

Generate {self.batch_size} new tickets in JSON format. Return a JSON object with a "tickets" array containing the ticket objects. Each ticket object should have these exact fields:
- ticket_id (string)
- customer_message (string)
- category (string)
- priority (string)
- customer_id (string)

Example format:
{{
  "tickets": [
    {{
      "ticket_id": "TICKET-001",
      "customer_message": "I cannot access my account",
      "category": "Authentication",
      "priority": "High",
      "customer_id": "CUST-001"
    }}
  ]
}}"""

        return prompt
    
    def generate_batch(self, examples: List[TicketExample]) -> List[Dict]:
        """Generate a batch of synthetic tickets using the Teacher model."""
        prompt = self.create_generation_prompt(examples)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",  # Use gpt-4o which supports json_object
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates realistic customer support tickets in JSON format. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=4000,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                logger.debug(f"Raw response: {content[:200]}...")
                
                if not content or not content.strip():
                    logger.warning("Empty response content")
                    continue
                
                # Handle both array and object responses
                if content.strip().startswith('['):
                    return json.loads(content)
                else:
                    data = json.loads(content)
                    # If response is wrapped in an object, extract the array
                    if isinstance(data, dict) and 'tickets' in data:
                        return data['tickets']
                    elif isinstance(data, dict) and any(isinstance(v, list) for v in data.values()):
                        for v in data.values():
                            if isinstance(v, list):
                                return v
                    elif isinstance(data, dict):
                        # If it's a single object, wrap it in a list
                        return [data]
                    else:
                        logger.warning(f"Unexpected response format: {content[:100]}...")
                        return []
                        
            except openai.RateLimitError:
                wait_time = (attempt + 1) * self.retry_delay
                logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                time.sleep(wait_time)
            except openai.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay)
        
        return []
    
    def validate_generated_ticket(self, ticket: Dict) -> bool:
        """Validate that a generated ticket meets our requirements."""
        # Handle different possible field names
        field_mappings = {
            'ticket_id': ['ticket_id', 'Ticket ID', 'ticketId', 'ticketID'],
            'customer_message': ['customer_message', 'Message', 'message'],
            'category': ['category', 'Category'],
            'priority': ['priority', 'Priority'],
            'customer_id': ['customer_id', 'Customer ID', 'customerId', 'customerID']
        }
        
        # Normalize the ticket fields
        normalized_ticket = {}
        for standard_field, possible_names in field_mappings.items():
            found = False
            for name in possible_names:
                if name in ticket and ticket[name]:
                    normalized_ticket[standard_field] = ticket[name]
                    found = True
                    break
            if not found:
                return False
        
        # Update the original ticket with normalized field names
        ticket.update(normalized_ticket)
        
        # Validate category and priority
        if normalized_ticket['category'] not in self.categories:
            return False
        if normalized_ticket['priority'] not in self.priorities:
            return False
        
        # Validate message length
        message_len = len(normalized_ticket['customer_message'])
        if message_len < 20 or message_len > 1000:
            return False
        
        return True
    
    def generate_synthetic_dataset(self, seed_file: str, output_file: str) -> None:
        """Generate the complete synthetic dataset."""
        logger.info("Starting synthetic data generation...")
        
        # Load seed examples
        examples = self.load_seed_examples(seed_file)
        
        # Calculate number of batches needed
        num_batches = (self.target_samples + self.batch_size - 1) // self.batch_size
        all_tickets = []
        
        for batch_num in range(num_batches):
            logger.info(f"Generating batch {batch_num + 1}/{num_batches}")
            
            # Generate batch
            batch_tickets = self.generate_batch(examples)
            
            # Validate and filter tickets
            valid_tickets = [ticket for ticket in batch_tickets if self.validate_generated_ticket(ticket)]
            
            if len(valid_tickets) < len(batch_tickets):
                logger.warning(f"Filtered out {len(batch_tickets) - len(valid_tickets)} invalid tickets")
            
            all_tickets.extend(valid_tickets)
            
            # Add some randomness to avoid rate limits
            time.sleep(random.uniform(1, 3))
            
            # Log progress
            logger.info(f"Generated {len(all_tickets)} tickets so far...")
            
            # Stop if we have enough tickets
            if len(all_tickets) >= self.target_samples:
                break
        
        # Trim to target size
        all_tickets = all_tickets[:self.target_samples]
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_tickets, f, indent=2, ensure_ascii=False)
        
        # Log statistics
        category_counts = {}
        priority_counts = {}
        for ticket in all_tickets:
            category_counts[ticket['category']] = category_counts.get(ticket['category'], 0) + 1
            priority_counts[ticket['priority']] = priority_counts.get(ticket['priority'], 0) + 1
        
        logger.info(f"Successfully generated {len(all_tickets)} synthetic tickets")
        logger.info(f"Category distribution: {category_counts}")
        logger.info(f"Priority distribution: {priority_counts}")
        logger.info(f"Dataset saved to: {output_file}")


def main():
    """Main function to run the synthetic data generation."""
    try:
        # Setup paths
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / "data"
        seed_file = data_dir / "seed_examples.json"
        output_file = data_dir / "synthetic_data.json"
        
        # Ensure data directory exists
        data_dir.mkdir(exist_ok=True)
        
        # Check if seed file exists
        if not seed_file.exists():
            logger.error(f"Seed examples file not found: {seed_file}")
            logger.info("Please create the seed_examples.json file first.")
            return
        
        # Initialize generator
        generator = SyntheticDataGenerator()
        
        # Generate synthetic data
        generator.generate_synthetic_dataset(str(seed_file), str(output_file))
        
        logger.info("Synthetic data generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to generate synthetic data: {e}")
        raise


if __name__ == "__main__":
    main()
