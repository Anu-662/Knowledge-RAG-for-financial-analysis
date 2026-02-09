"""
Entity Extractor - Phase 2
Extracts companies, people, events, and metrics from financial articles using GPT-4
"""

import os
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import time
from dotenv import load_dotenv

load_dotenv()

# You'll need: pip install openai
try:
    from openai import OpenAI
except ImportError:
    print("❌ OpenAI not installed. Run: pip install openai")
    exit(1)


class EntityExtractor:
    """Extract entities from financial articles using GPT-4"""
    
    def __init__(self, data_dir: str = None):
        # Auto-detect project root
        if data_dir is None:
            script_dir = Path(__file__).parent
            if script_dir.name == "kg_builder":
                project_root = script_dir.parent.parent
            else:
                project_root = Path.cwd()
            data_dir = project_root / "data"
        else:
            data_dir = Path(data_dir)
        
        self.data_dir = data_dir
        self.processed_dir = data_dir / "processed"
        self.entities_dir = data_dir / "entities"
        self.entities_dir.mkdir(exist_ok=True)
        
        # OpenAI setup
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ OPENAI_API_KEY not found!\n"
                "Add to .env file: OPENAI_API_KEY=sk-your-key-here\n"
                "Get key from: https://platform.openai.com/api-keys"
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # Cheaper, faster, still great for extraction
        
        # Track costs
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def create_extraction_prompt(self, article: Dict) -> str:
        """Create prompt for entity extraction"""
        
        prompt = f"""Extract financial entities from this article. Be precise and only extract entities explicitly mentioned.

Article Title: {article['title']}
Article Text: {article['full_text']}

Extract and return ONLY valid JSON (no markdown, no explanation):

{{
  "companies": [
    {{"name": "Apple Inc.", "ticker": "AAPL", "role": "primary subject"}},
    {{"name": "Microsoft", "ticker": "MSFT", "role": "competitor"}}
  ],
  "people": [
    {{"name": "Tim Cook", "role": "CEO", "company": "Apple Inc."}}
  ],
  "sectors": ["Technology", "Consumer Electronics"],
  "events": [
    {{"type": "earnings_report", "description": "Q4 earnings beat expectations", "date": "2024-01-28"}}
  ],
  "metrics": [
    {{"type": "revenue", "value": "119.6B", "unit": "USD", "period": "Q4 2024"}},
    {{"type": "stock_price_change", "value": "5.2", "unit": "percent"}}
  ],
  "locations": ["Cupertino", "California"],
  "products": ["iPhone 15", "Vision Pro"],
  "relationships": [
    {{"entity1": "Apple Inc.", "relationship": "competes_with", "entity2": "Microsoft"}},
    {{"entity1": "Apple Inc.", "relationship": "operates_in", "entity2": "Technology"}}
  ]
}}

Rules:
- Only extract entities EXPLICITLY mentioned in the text
- For companies: Include full name and ticker if available
- For events: Extract type, description, date if mentioned
- For metrics: Extract numerical values with context
- For relationships: Identify competitive, partnership, or hierarchical relationships
- If ticker not mentioned, use empty string
- If date not mentioned, use null
- Return ONLY the JSON, no other text
"""
        return prompt
    
    def extract_entities(self, article: Dict, retry_count: int = 3) -> Dict:
        """Extract entities from a single article using GPT-4"""
        
        prompt = self.create_extraction_prompt(article)
        
        for attempt in range(retry_count):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a financial entity extraction expert. Extract entities precisely and return only valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.0,  # Deterministic
                    max_tokens=1000
                )
                
                # Track usage
                self.total_tokens += response.usage.total_tokens
                # GPT-4o-mini pricing: $0.15/1M input, $0.60/1M output
                input_cost = (response.usage.prompt_tokens / 1_000_000) * 0.15
                output_cost = (response.usage.completion_tokens / 1_000_000) * 0.60
                self.total_cost += (input_cost + output_cost)
                
                # Parse response
                content = response.choices[0].message.content.strip()
                
                # Remove markdown code blocks if present
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()
                
                entities = json.loads(content)
                
                # Add metadata
                entities["article_id"] = article["id"]
                entities["article_title"] = article["title"]
                entities["extraction_date"] = datetime.now().isoformat()
                
                return entities
                
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON parse error (attempt {attempt + 1}/{retry_count}): {str(e)[:50]}")
                if attempt == retry_count - 1:
                    return self._create_empty_entities(article["id"], article["title"])
                time.sleep(1)
                
            except Exception as e:
                print(f"  ❌ Error extracting from article {article['id']}: {str(e)[:100]}")
                if attempt == retry_count - 1:
                    return self._create_empty_entities(article["id"], article["title"])
                time.sleep(2)
        
        return self._create_empty_entities(article["id"], article["title"])
    
    def _create_empty_entities(self, article_id: str, title: str) -> Dict:
        """Create empty entity structure for failed extractions"""
        return {
            "article_id": article_id,
            "article_title": title,
            "companies": [],
            "people": [],
            "sectors": [],
            "events": [],
            "metrics": [],
            "locations": [],
            "products": [],
            "relationships": [],
            "extraction_date": datetime.now().isoformat(),
            "extraction_failed": True
        }
    
    def process_all_articles(self, limit: int = None, start_from: int = 0):
        """Process all articles and extract entities"""
        
        # Load all articles
        article_files = sorted(list(self.processed_dir.glob("*.json")))
        
        if limit:
            article_files = article_files[start_from:start_from + limit]
        else:
            article_files = article_files[start_from:]
        
        total = len(article_files)
        
        print("🧠 ENTITY EXTRACTION - PHASE 2")
        print("="*60)
        print(f"📊 Processing {total} articles")
        print(f"🤖 Model: {self.model}")
        print(f"💰 Estimated cost: ${(total * 0.015):.2f} (worst case)\n")
        
        all_entities = []
        failed_count = 0
        
        for i, article_file in enumerate(article_files, 1):
            # Load article
            with open(article_file) as f:
                article = json.load(f)
            
            print(f"[{i}/{total}] {article['title'][:50]}...", end=" ")
            
            # Extract entities
            entities = self.extract_entities(article)
            
            if entities.get("extraction_failed"):
                failed_count += 1
                print("❌ FAILED")
            else:
                # Count entities
                entity_count = sum([
                    len(entities.get("companies", [])),
                    len(entities.get("people", [])),
                    len(entities.get("events", []))
                ])
                print(f"✓ {entity_count} entities")
            
            all_entities.append(entities)
            
            # Save individual entity file
            entity_file = self.entities_dir / f"{article['id']}_entities.json"
            with open(entity_file, 'w') as f:
                json.dump(entities, f, indent=2)
            
            # Progress update every 10 articles
            if i % 10 == 0:
                print(f"\n💰 Cost so far: ${self.total_cost:.4f} | Tokens: {self.total_tokens:,}\n")
            
            # Rate limiting (5 requests per second for gpt-4o-mini)
            time.sleep(0.2)
        
        # Save consolidated entities
        self._save_consolidated_entities(all_entities)
        
        # Print summary
        self._print_summary(all_entities, failed_count)
        
        return all_entities
    
    def _save_consolidated_entities(self, all_entities: List[Dict]):
        """Save consolidated entity lists"""
        
        # Aggregate all entities
        all_companies = []
        all_people = []
        all_sectors = set()
        all_events = []
        all_relationships = []
        
        for entities in all_entities:
            all_companies.extend(entities.get("companies", []))
            all_people.extend(entities.get("people", []))
            all_sectors.update(entities.get("sectors", []))
            all_events.extend(entities.get("events", []))
            all_relationships.extend(entities.get("relationships", []))
        
        # Deduplicate companies by name
        unique_companies = {}
        for company in all_companies:
            name = company.get("name", "").strip()
            if name and name not in unique_companies:
                unique_companies[name] = company
        
        # Save consolidated file
        consolidated = {
            "extraction_date": datetime.now().isoformat(),
            "total_articles": len(all_entities),
            "total_companies": len(unique_companies),
            "total_people": len(all_people),
            "total_sectors": len(all_sectors),
            "total_events": len(all_events),
            "total_relationships": len(all_relationships),
            "unique_companies": list(unique_companies.values()),
            "sectors": sorted(list(all_sectors)),
            "sample_people": all_people[:50],  # First 50 people
            "sample_events": all_events[:50],
            "sample_relationships": all_relationships[:50],
            "cost": {
                "total_tokens": self.total_tokens,
                "total_cost_usd": round(self.total_cost, 4)
            }
        }
        
        output_file = self.entities_dir / "consolidated_entities.json"
        with open(output_file, 'w') as f:
            json.dump(consolidated, f, indent=2)
        
        print(f"\n💾 Saved consolidated entities to: {output_file}")
    
    def _print_summary(self, all_entities: List[Dict], failed_count: int):
        """Print extraction summary"""
        
        # Count entities
        total_companies = sum(len(e.get("companies", [])) for e in all_entities)
        total_people = sum(len(e.get("people", [])) for e in all_entities)
        total_events = sum(len(e.get("events", [])) for e in all_entities)
        total_metrics = sum(len(e.get("metrics", [])) for e in all_entities)
        total_relationships = sum(len(e.get("relationships", [])) for e in all_entities)
        
        # Unique companies
        all_company_names = set()
        for entities in all_entities:
            for company in entities.get("companies", []):
                name = company.get("name", "").strip()
                if name:
                    all_company_names.add(name)
        
        # All sectors
        all_sectors = set()
        for entities in all_entities:
            all_sectors.update(entities.get("sectors", []))
        
        print("\n" + "="*60)
        print("📊 EXTRACTION SUMMARY")
        print("="*60)
        print(f"✅ Successfully processed: {len(all_entities) - failed_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"\n📈 ENTITIES EXTRACTED:")
        print(f"  Companies (total mentions): {total_companies}")
        print(f"  Companies (unique): {len(all_company_names)}")
        print(f"  People: {total_people}")
        print(f"  Events: {total_events}")
        print(f"  Metrics: {total_metrics}")
        print(f"  Relationships: {total_relationships}")
        print(f"  Sectors found: {len(all_sectors)}")
        print(f"\n💰 COST:")
        print(f"  Total tokens: {self.total_tokens:,}")
        print(f"  Total cost: ${self.total_cost:.4f}")
        print(f"\n✅ Ready for Neo4j import!")


if __name__ == "__main__":
    import sys
    
    extractor = EntityExtractor()
    
    # Parse command line args
    limit = None
    start = 0
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            limit = 10
            print("🧪 TEST MODE: Processing first 10 articles\n")
        elif sys.argv[1].isdigit():
            limit = int(sys.argv[1])
            print(f"📊 Processing {limit} articles\n")
    
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        start = int(sys.argv[2])
        print(f"⏩ Starting from article {start}\n")
    
    # Run extraction
    extractor.process_all_articles(limit=limit, start_from=start)
