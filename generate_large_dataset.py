#!/usr/bin/env python3
"""
Efficient Large Dataset Generation for LLM Data Factory

This script generates 1,200+ synthetic customer support tickets efficiently
using parallel processing and optimized API calls.
"""

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('large_dataset_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EfficientDataGenerator:
    """Efficient generator for large synthetic datasets."""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.categories = ["Authentication", "Technical", "Billing", "Feature Request", "General"]
        self.priorities = ["Low", "Medium", "High", "Critical"]
        self.target_samples = 1200
        self.samples_per_batch = 20  # Generate more per API call
        self.max_workers = 3  # Parallel workers
        self.all_tickets = []
        self.lock = threading.Lock()
        
    def load_seed_examples(self, file_path: str) -> List[Dict]:
        """Load seed examples from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_generation_prompt(self, examples: List[Dict]) -> str:
        """Create an efficient prompt for batch generation."""
        # Take first 5 examples to keep prompt concise
        sample_examples = examples[:5]
        examples_text = "\n\n".join([
            f"Ticket: {ex['customer_message']}\n"
            f"Category: {ex['category']}\n"
            f"Priority: {ex['priority']}"
            for ex in sample_examples
        ])
        
        return f"""Generate {self.samples_per_batch} diverse, realistic customer support tickets in JSON format.

Categories: {', '.join(self.categories)}
Priorities: {', '.join(self.priorities)}

Example tickets:
{examples_text}

Generate tickets covering various issues like:
- Login/authentication problems
- Technical bugs and crashes  
- Billing and payment issues
- Feature requests and suggestions
- General questions and how-to inquiries

Return JSON with "tickets" array. Each ticket needs:
- ticket_id: "TICKET-XXXXX" 
- customer_message: realistic detailed message (50-200 words)
- category: one of the categories above
- priority: one of the priorities above  
- customer_id: "CUST-XXXXX"

Make messages diverse and realistic - vary the tone, length, technical detail, and urgency."""

    def generate_batch(self, examples: List[Dict], batch_num: int) -> List[Dict]:
        """Generate a single batch of tickets."""
        prompt = self.create_generation_prompt(examples)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at generating realistic customer support tickets. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,  # Higher temperature for more diversity
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            tickets = data.get('tickets', [])
            if not tickets:
                logger.warning(f"Batch {batch_num}: No tickets in response")
                return []
            
            # Validate and fix tickets
            valid_tickets = []
            for i, ticket in enumerate(tickets):
                if self.validate_and_fix_ticket(ticket, batch_num, i):
                    valid_tickets.append(ticket)
            
            logger.info(f"Batch {batch_num}: Generated {len(valid_tickets)}/{len(tickets)} valid tickets")
            return valid_tickets
            
        except Exception as e:
            logger.error(f"Batch {batch_num}: Error generating tickets: {e}")
            return []
    
    def validate_and_fix_ticket(self, ticket: Dict, batch_num: int, ticket_idx: int) -> bool:
        """Validate and fix a ticket."""
        # Fix ticket ID
        if 'ticket_id' not in ticket or not ticket['ticket_id']:
            ticket['ticket_id'] = f"TICKET-{batch_num:03d}{ticket_idx:02d}"
        
        # Fix customer ID  
        if 'customer_id' not in ticket or not ticket['customer_id']:
            ticket['customer_id'] = f"CUST-{batch_num:03d}{ticket_idx:02d}"
        
        # Validate required fields
        required_fields = ['customer_message', 'category', 'priority']
        for field in required_fields:
            if field not in ticket or not ticket[field]:
                return False
        
        # Validate category and priority
        if ticket['category'] not in self.categories:
            # Try to map to valid category
            message = ticket['customer_message'].lower()
            if any(word in message for word in ['login', 'password', 'account', 'access']):
                ticket['category'] = 'Authentication'
            elif any(word in message for word in ['crash', 'bug', 'error', 'broken']):
                ticket['category'] = 'Technical'
            elif any(word in message for word in ['billing', 'payment', 'charge', 'subscription']):
                ticket['category'] = 'Billing'
            elif any(word in message for word in ['feature', 'add', 'improve', 'suggestion']):
                ticket['category'] = 'Feature Request'
            else:
                ticket['category'] = 'General'
        
        if ticket['priority'] not in self.priorities:
            ticket['priority'] = random.choice(self.priorities)
        
        # Validate message length
        if len(ticket['customer_message']) < 20:
            return False
            
        return True
    
    def worker_function(self, examples: List[Dict], batch_numbers: List[int]) -> List[Dict]:
        """Worker function for parallel generation."""
        worker_tickets = []
        for batch_num in batch_numbers:
            batch_tickets = self.generate_batch(examples, batch_num)
            worker_tickets.extend(batch_tickets)
            
            # Add some delay to avoid rate limits
            time.sleep(random.uniform(0.5, 1.5))
        
        return worker_tickets
    
    def generate_large_dataset(self, seed_file: str, output_file: str):
        """Generate the complete large dataset efficiently."""
        logger.info(f"Starting generation of {self.target_samples} synthetic tickets...")
        
        # Load seed examples
        examples = self.load_seed_examples(seed_file)
        logger.info(f"Loaded {len(examples)} seed examples")
        
        # Calculate batches needed
        batches_needed = (self.target_samples + self.samples_per_batch - 1) // self.samples_per_batch
        logger.info(f"Will generate {batches_needed} batches of {self.samples_per_batch} tickets each")
        
        # Distribute batches across workers
        batch_numbers = list(range(1, batches_needed + 1))
        chunks = [batch_numbers[i::self.max_workers] for i in range(self.max_workers)]
        
        all_tickets = []
        
        # Use parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(self.worker_function, examples, chunk): chunk 
                for chunk in chunks if chunk
            }
            
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    tickets = future.result()
                    all_tickets.extend(tickets)
                    logger.info(f"Completed chunk, total tickets so far: {len(all_tickets)}")
                except Exception as e:
                    logger.error(f"Chunk failed: {e}")
        
        # Trim to target size and shuffle
        random.shuffle(all_tickets)
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
    """Main function."""
    try:
        script_dir = Path(__file__).parent
        data_dir = script_dir / "data"
        seed_file = data_dir / "seed_examples.json"
        output_file = data_dir / "large_synthetic_data.json"
        
        data_dir.mkdir(exist_ok=True)
        
        if not seed_file.exists():
            logger.error(f"Seed examples file not found: {seed_file}")
            return
        
        generator = EfficientDataGenerator()
        generator.generate_large_dataset(str(seed_file), str(output_file))
        
        logger.info("Large synthetic dataset generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to generate large dataset: {e}")
        raise

if __name__ == "__main__":
    main()
