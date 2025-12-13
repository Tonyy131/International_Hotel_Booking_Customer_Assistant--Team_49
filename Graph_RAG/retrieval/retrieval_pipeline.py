from typing import Dict, Any, List
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
            print(entities,"mizpppy")
            baseline_results, executed_cypher = self.baseline.retrieve(intent, entities, limit=limit)
            print(baseline_results, "mizpppp")
        else:
            baseline_results = []
            executed_cypher = ""

        embedding_results = []
        if user_embeddings:
            # For hotel-oriented intents we search hotels + reviews
            rating_filter = entities.get("rating_filter") if entities else None
            embedding_results = self.embed.sem_search_hotels(user_query, entities, top_k=limit, rating_filter=rating_filter, intent=intent)

        combined = self._merge_results(baseline_results, embedding_results)

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
        Merges results while preserving 'origin_country' and 'destination_country' 
        for visa queries, alongside standard hotel data.
        """
        hotels = []
        visa_info = []
        seen_ids = set()

        def process_list(result_list, source_name):
            for r in result_list or []:
                
                if "visa_type" in r:
                    visa_info.append({**r, "source": source_name})
                    continue 

                hid = r.get("hotel_id") or r.get("hotel", {}).get("hotel_id") if isinstance(r.get("hotel"), dict) else None
                name = r.get("name") or (r.get("h") and r.get("h").get("name"))
                
                key = hid or name
                if not key:
                    continue
                
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                hotels.append({**r, "source": source_name})

        process_list(baseline, "baseline")
        process_list(embedding, "embedding")

        return {"hotels": hotels, "reviews": [], "visa_info": visa_info}
    

    def _build_context_text(self, combined: Dict[str, Any]) -> str:
        parts = []
        
        # --- Handle Visa Context ---
        visa_list = combined.get("visa_info", [])
        if visa_list:
            parts.append("--- Visa Information ---")
            for v in visa_list:
                # We extract the specific country keys here
                origin = v.get("origin_country", "Unknown Origin")
                dest = v.get("destination_country", "Unknown Destination")
                req = v.get("visa_type", "Unknown Requirement")
                
                parts.append(f"• Travel from {origin} to {dest}: {req}")
            parts.append("")

        # --- Handle Hotel Context (Existing logic) ---
        hotels = combined.get("hotels", [])
        if hotels:
            parts.append("--- Retrieved Hotels ---")
            for h in hotels:
                name = h.get("name")
                avg_score = h.get("average_reviews_score")
                line = f"- {name}"
                if avg_score:
                    line += f" (Score: {avg_score})"
                parts.append(line)

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
