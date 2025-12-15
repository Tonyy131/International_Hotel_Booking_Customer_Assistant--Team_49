from typing import Dict, Any, List, Optional
from retrieval.baseline_retriever import BaselineRetriever
from retrieval.embedding_retriever import EmbeddingRetriever
from preprocessing.entity_extractor import EntityExtractor
from preprocessing.preprocess_intent import classify_user_intent
from neo4j_connector import Neo4jConnector

class RetrievalPipeline:
    """
    Orchestrates baseline + embedding retrieval and merges results into a single context
    structure suitable for feeding to the LLM prompt builder.
    """
    def __init__(self, neo4j_connector: Neo4jConnector = None, model_name: str = "minilm"):
        connector = neo4j_connector or Neo4jConnector()
        self.baseline = BaselineRetriever(connector)
        self.model_name = model_name
        self.embed = EmbeddingRetriever(connector, model_name=model_name)
        
    def retrieve(self, intent: str, entities: Dict[str, Any], user_query: str, user_embeddings: bool = True, limit: int = 10, user_baseline: bool = True) -> Dict[str, Any]:
        if user_baseline:
            baseline_results, executed_cypher = self.baseline.retrieve(intent, entities, limit=limit)
        else:
            baseline_results = []
            executed_cypher = ""

        embedding_results = []
        if user_embeddings:
            # For hotel-oriented intents we search hotels + reviews
            rating_filter = entities.get("rating_filter") if entities else None
            embedding_results = self.embed.sem_search_hotels(user_query, entities, top_k=limit, rating_filter=rating_filter, intent=intent)

        combined = self._merge_results(baseline_results, embedding_results)
        print("combinedddddddddd", combined)

        # Build a textual context summary (simple)
        context_text = self._build_context_text(combined)

        return {
            "baseline": baseline_results,
            "embeddings": embedding_results,
            "combined": combined,
            "context_text": context_text,
            "cypher_query": executed_cypher
        }


    def _merge_results(self, baseline: List[Dict], embedding: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Merges results, handles 'hotel_name' alias, deduplicates, and strips vectors.
        """
        hotels = []
        visa_info = []
        others = []
        seen_ids = set()

        def process_list(result_list, source_name):
            for r in result_list or []:
                
                # --- 1. Handle Visa ---
                if "visa_type" in r:
                    visa_info.append({**r, "source": source_name})
                    continue 

                # --- 2. Normalize Hotel Data ---
                if "h" in r and isinstance(r.get("h"), dict):
                    # Already nested (Embedding Retriever)
                    hotel_node = r["h"]
                    item_container = r.copy()
                else:
                    # Flat (Baseline Retriever) -> Nest it
                    hotel_node = r.copy() # Copy to avoid mutating original
                    
                    # FIX: Map 'hotel_name' to 'name' so it is recognized as a hotel
                    if "hotel_name" in hotel_node and "name" not in hotel_node:
                        hotel_node["name"] = hotel_node["hotel_name"]

                    item_container = {
                        "h": hotel_node,
                        "source": source_name,
                        "city_name": r.get("city_name") or r.get("city"),
                        "country_name": r.get("country_name") or r.get("country"),
                        "review_texts": r.get("review_texts", [])
                    }

                # --- 3. Identify & Deduplicate ---
                hid = hotel_node.get("hotel_id")
                name = hotel_node.get("name") # Now valid for Baseline too
                
                key = hid or name
                
                if key:
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                    
                    # Remove vectors
                    hotel_node.pop("embedding_minilm", None)
                    hotel_node.pop("embedding_bge", None)
                    
                    hotels.append(item_container)
                else:
                    # Only generic info goes here
                    others.append({**r, "source": source_name})

        process_list(baseline, "baseline")
        process_list(embedding, "embedding")

        return {"hotels": hotels, "visa_info": visa_info, "others": others}
   
    def _build_context_text(self, combined: Dict[str, Any]) -> str:
        parts = []
        
        # --- 1. Handle Visa Context ---
        visa_list = combined.get("visa_info", [])
        if visa_list:
            parts.append("--- Visa Information ---")
            for v in visa_list:
                origin = v.get("origin_country", "Unknown Origin")
                dest = v.get("destination_country", "Unknown Destination")
                req = v.get("visa_type", "Unknown Requirement")
                parts.append(f"• Travel from {origin} to {dest}: {req}")
            parts.append("")

        # --- 2. Handle Hotel Context ---
        hotels = combined.get("hotels", [])
        if hotels:
            parts.append("--- Retrieved Hotels ---")
            for item in hotels:
                # Unpack the nested 'h' dictionary
                hotel_node = item.get("h", {})
                
                # --- A. Extract Location ---
                # Fallback to internal keys if the wrapper keys are empty
                city = item.get("city_name") or hotel_node.get("city") or "Unknown City"
                country = item.get("country_name") or hotel_node.get("country") or "Unknown Country"
                
                # --- B. Extract Basic Info ---
                name = hotel_node.get("name", "Unnamed Hotel")

                star_rating = hotel_node.get("star_rating")
                
                avg_score = (hotel_node.get("average_reviews_score") or 
                             hotel_node.get("total_avg_score"))
                
                line = f"• {name} (Located in {city}, {country})"

                if star_rating:
                    line += f" | {star_rating} Stars"


                if avg_score:
                    line += f" | Global Rating: {float(avg_score):.1f}/10"

                cat_scores = []
                categories = [
                    ("avg_score_cleanliness", "Cleanliness"),
                    ("avg_score_comfort", "Comfort"),
                    ("avg_score_facilities", "Facilities"),
                    ("avg_score_staff", "Staff")
                ]

                for key, label in categories:
                    val = hotel_node.get(key)
                    # Fallback to base score if dynamic average is missing/None
                    if val is None:
                        val = hotel_node.get(f"{key.replace('avg_score_', '')}_base")
                        
                    if val is not None:
                        cat_scores.append(f"{label}: {float(val):.1f}")
                
                if cat_scores:
                    line += " | [" + ", ".join(cat_scores) + "]"

                # --- C. Extract Traveler Type Scores ---
                traveller_scores = []
                for key, value in hotel_node.items():
                    if key.startswith("avg_score_") and isinstance(value, (int, float)):
                        readable_type = key.replace("avg_score_", "").replace("_", " ").title()
                        traveller_scores.append(f"{readable_type}: {value:.1f}")
                
                if traveller_scores:
                    line += " | Ratings: [" + ", ".join(traveller_scores) + "]"

                parts.append(line)

                # --- D. Extract Reviews ---
                # FIX 2: Check BOTH review sources
                # Source 1: List of reviews (Embedding Retriever)
                reviews_list = item.get("review_texts", [])
                
                # Source 2: Single latest review (Baseline Retriever)
                latest_review = hotel_node.get("latest_review_text")
                
                # Combine them safely
                final_reviews = []
                if reviews_list:
                    final_reviews.extend(reviews_list)
                if latest_review and latest_review not in final_reviews:
                    final_reviews.append(latest_review)

                if final_reviews:
                    clean_reviews = [str(r).replace("\n", " ").strip() for r in final_reviews if r]
                    # Show up to 2 reviews
                    for review in clean_reviews[:2]:
                        parts.append(f"    - Review: \"{review[:150]}...\"")
            
            parts.append("")

        # --- 3. Handle 'Others' ---
        others = combined.get("others", [])
        if others:
            parts.append("--- Other Information ---")
            for item in others:
                if isinstance(item, dict):
                    # Smart formatting for miscellaneous dicts
                    clean_pairs = []
                    for k, v in item.items():
                        if k not in ["source", "embedding_minilm", "embedding_bge"] and v is not None:
                            clean_pairs.append(f"{k}: {v}")
                    if clean_pairs:
                        parts.append("- " + " | ".join(clean_pairs))
                else:
                    parts.append(f"- {str(item)}")
            parts.append("")

        if not parts:
            return "No relevant information found."

        return "\n".join(parts)

    def safe_retrieve(
        self,
        query: str,
        limit: int = 10,
        user_embeddings: bool = True,
        use_llm: bool = True,
        user_baseline: bool = True
    ) -> Dict[str, Any]:
        """
        Production-safe retrieval wrapper for the final chatbot.

        - Uses extracted intent
        - Never crashes
        - Normalizes return format
        - Fallbacks if something goes wrong
        - Suitable for chatbot + experiments with real pipeline behavior
        """

        # 1) Extract entities
        extractor = EntityExtractor()
        entities = extractor.extract(query, use_llm=use_llm)

        # 2) Classify intent
        try:
            intent_info = classify_user_intent(query)
            intent = intent_info.get("intent")
            if not intent:
                intent = "hotel_search"     # fallback
        except Exception:
            intent = "hotel_search"         # fallback

        # 3) Run the actual retrieval
        try:
            print("entitiessss", entities)
            results = self.retrieve(
                intent=intent,
                entities=entities,
                user_query=query,
                user_embeddings=user_embeddings,
                limit=limit,
                user_baseline=user_baseline
            )
        except Exception as e:
            # Critical failure → return safe empty structure
            return {
                "intent": intent,
                "entities": entities,
                "baseline": [],
                "embeddings": [],
                "combined": {"hotels": [], "reviews": []},
                "context_text": f"[Retrieval Error: {str(e)}]",
                "cypher_query": f"// Error generating query: {str(e)}"
            }
        
        # 4) Normalize structure and return clean results
        return {
            "intent": intent,
            "entities": entities,
            "baseline": results.get("baseline", []),
            "embeddings": results.get("embeddings", []),
            "combined": results.get("combined", {}),
            "context_text": results.get("context_text", ""),
            "cypher_query": results.get("cypher_query", "// No Cypher executed")
        }
