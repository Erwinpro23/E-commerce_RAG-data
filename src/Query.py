import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class RAGRetriever:
    """
    RAG Retrieval System for E-commerce Data

    Loads FAISS index and metadata, provides semantic search functionality.
    """

    def __init__(self, faiss_index_path: str = "faiss_index.bin",
                 metadata_path: str = "metadata.pkl",
                 model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the RAG retriever.

        Args:
            faiss_index_path: Path to FAISS index file
            metadata_path: Path to metadata pickle file
            model_name: Sentence transformer model name
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(faiss_index_path)
            print(f"✅ FAISS index loaded: {self.index.ntotal} vectors")

            # Load metadata
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
            self.texts = data["texts"]
            self.metadatas = data["metadatas"]
            print(f"✅ Metadata loaded: {len(self.texts)} texts, {len(self.metadatas)} metadatas")

            # Load embedding model
            self.model = SentenceTransformer(model_name)
            print(f"✅ Embedding model loaded: {model_name}")

        except Exception as e:
            raise Exception(f"Error initializing RAGRetriever: {str(e)}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of dictionaries with 'text', 'metadata', 'distance', 'rank'
        """
        try:
            # Encode query
            query_embedding = self.model.encode([query]).astype("float32")

            # Search FAISS
            distances, indices = self.index.search(query_embedding, top_k)

            # Prepare results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.texts):  # Safety check
                    result = {
                        "text": self.texts[idx],
                        "metadata": self.metadatas[idx],
                        "distance": float(distances[0][i]),
                        "rank": i + 1
                    }
                    results.append(result)

            # Sort by distance descending (high to low, worst to best similarity)
            results.sort(key=lambda x: x['distance'], reverse=True)

            return results

        except Exception as e:
            raise Exception(f"Error during retrieval: {str(e)}")

    def _calculate_fairness_scores(self) -> Dict[str, float]:
        """
        Calculate fairness scores based on group distributions in the dataset.

        Returns:
            Dictionary mapping group keys to fairness scores
        """
        from collections import Counter

        # Calculate distribution of different granularities
        granularity_counts = Counter()
        region_counts = Counter()
        category_counts = Counter()
        segment_counts = Counter()

        for meta in self.metadatas:
            granularity_counts[meta.get('granularity', 'unknown')] += 1
            if 'Region' in meta:
                region_counts[meta['Region']] += 1
            if 'Category' in meta:
                category_counts[meta['Category']] += 1
            if 'Segment' in meta:
                segment_counts[meta['Segment']] += 1

        # Calculate fairness scores (inverse frequency - rarer groups get higher scores)
        total_docs = len(self.metadatas)

        fairness_scores = {}

        # Granularity fairness
        for gran, count in granularity_counts.items():
            fairness_scores[f"granularity_{gran}"] = 1.0 / (count / total_docs)

        # Region fairness
        for region, count in region_counts.items():
            fairness_scores[f"region_{region}"] = 1.0 / (count / total_docs)

        # Category fairness
        for cat, count in category_counts.items():
            fairness_scores[f"category_{cat}"] = 1.0 / (count / total_docs)

        # Segment fairness
        for seg, count in segment_counts.items():
            fairness_scores[f"segment_{seg}"] = 1.0 / (count / total_docs)

        return fairness_scores

    def _calculate_redundancy_penalty(self, selected_results: List[Dict], candidate: Dict) -> float:
        """
        Calculate redundancy penalty based on similarity to already selected results.

        Args:
            selected_results: Already selected results
            candidate: Candidate result to evaluate

        Returns:
            Redundancy penalty score (higher = more redundant)
        """
        if not selected_results:
            return 0.0

        # Simple redundancy based on granularity overlap
        candidate_gran = candidate['metadata'].get('granularity', '')
        candidate_region = candidate['metadata'].get('Region', '')
        candidate_category = candidate['metadata'].get('Category', '')

        redundancy_score = 0.0

        for selected in selected_results:
            selected_gran = selected['metadata'].get('granularity', '')
            selected_region = selected['metadata'].get('Region', '')
            selected_category = selected['metadata'].get('Category', '')

            # Penalty for same granularity
            if candidate_gran == selected_gran:
                redundancy_score += 0.3

            # Penalty for same region
            if candidate_region == selected_region and candidate_region:
                redundancy_score += 0.2

            # Penalty for same category
            if candidate_category == selected_category and candidate_category:
                redundancy_score += 0.2

        return min(redundancy_score, 1.0)  # Cap at 1.0

    def fairness_aware_rerank(self, candidates: List[Dict], top_k: int = 5,
                             alpha: float = 0.7, beta: float = 0.2, gamma: float = 0.1) -> List[Dict]:
        """
        Fairness-aware reranking using greedy selection algorithm.

        Args:
            candidates: Candidate results from vector search (should be 20-50)
            top_k: Final number of results to return
            alpha: Weight for relevance score
            beta: Weight for fairness score
            gamma: Weight for redundancy penalty

        Returns:
            Reranked results with fairness considerations
        """
        if not candidates:
            return []

        fairness_scores = self._calculate_fairness_scores()

        selected = []
        remaining = candidates.copy()

        while len(selected) < top_k and remaining:
            best_score = -float('inf')
            best_candidate = None
            best_idx = -1

            for i, candidate in enumerate(remaining):
                # Relevance score (lower distance = higher relevance)
                relevance_score = 1.0 / (1.0 + candidate['distance'])

                # Fairness score
                fairness_score = 0.0
                meta = candidate['metadata']

                # Check different group memberships
                gran_key = f"granularity_{meta.get('granularity', 'unknown')}"
                fairness_score += fairness_scores.get(gran_key, 0.1)

                if 'Region' in meta:
                    region_key = f"region_{meta['Region']}"
                    fairness_score += fairness_scores.get(region_key, 0.1)

                if 'Category' in meta:
                    cat_key = f"category_{meta['Category']}"
                    fairness_score += fairness_scores.get(cat_key, 0.1)

                if 'Segment' in meta:
                    seg_key = f"segment_{meta['Segment']}"
                    fairness_score += fairness_scores.get(seg_key, 0.1)

                # Normalize fairness score
                fairness_score = min(fairness_score / 4.0, 1.0)

                # Redundancy penalty
                redundancy_penalty = self._calculate_redundancy_penalty(selected, candidate)

                # Final score
                final_score = (alpha * relevance_score +
                             beta * fairness_score -
                             gamma * redundancy_penalty)

                if final_score > best_score:
                    best_score = final_score
                    best_candidate = candidate
                    best_idx = i

            if best_candidate:
                selected.append(best_candidate)
                remaining.pop(best_idx)

        # Update ranks
        for i, result in enumerate(selected):
            result['rank'] = i + 1
            result['final_score'] = best_score if i == len(selected) - 1 else None

        return selected

    def retrieve_with_fairness(self, query: str, top_k: int = 5,
                              candidate_pool_size: int = 30) -> List[Dict[str, Any]]:
        """
        Retrieve with fairness-aware reranking.

        Args:
            query: Search query string
            top_k: Final number of results to return
            candidate_pool_size: Size of candidate pool for reranking (20-50 recommended)

        Returns:
            Fairness-aware reranked results
        """
        try:
            # Get larger candidate pool
            candidates = self.retrieve(query, top_k=max(candidate_pool_size, top_k * 2))

            # Apply fairness-aware reranking
            reranked = self.fairness_aware_rerank(candidates, top_k=top_k)

            return reranked

        except Exception as e:
            raise Exception(f"Error during fairness-aware retrieval: {str(e)}")

    def pre_filter_by_metadata(self, query: str, filter_criteria: Dict[str, Any],
                              top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Pre-filter documents by metadata before vector search.

        Args:
            query: Search query
            filter_criteria: Dict of metadata filters (e.g., {'Region': 'South'})
            top_k: Number of results to return

        Returns:
            Filtered results
        """
        # First get all candidates
        all_candidates = self.retrieve(query, top_k=self.index.ntotal)

        # Apply metadata filters
        filtered = []
        for candidate in all_candidates:
            meta = candidate['metadata']
            match = True

            for key, value in filter_criteria.items():
                if key not in meta or meta[key] != value:
                    match = False
                    break

            if match:
                filtered.append(candidate)

        # Return top_k filtered results
        return filtered[:top_k]

    def post_filter_diversity_check(self, results: List[Dict], max_region_ratio: float = 0.6) -> List[Dict]:
        """
        Post-filter to ensure diversity (e.g., no more than 60% from same region).

        Args:
            results: Results to check
            max_region_ratio: Maximum ratio allowed for any single region

        Returns:
            Filtered results maintaining diversity
        """
        if not results:
            return results

        from collections import Counter

        # Count regions in results
        regions = [r['metadata'].get('Region', 'Unknown') for r in results]
        region_counts = Counter(regions)

        filtered = []
        current_counts = Counter()

        for result in results:
            region = result['metadata'].get('Region', 'Unknown')

            # Check if adding this result would exceed max ratio
            if current_counts[region] < max_region_ratio * len(filtered) or len(filtered) == 0:
                filtered.append(result)
                current_counts[region] += 1

        return filtered

    def print_results(self, results: List[Dict[str, Any]], query: str = ""):
        """
        Print retrieval results in a formatted way.

        Args:
            results: List of retrieval results
            query: Original query (optional)
        """
        if query:
            print(f"\n🔍 Query: {query}")

        print(f"📊 Top {len(results)} results (sorted by distance descending - worst to best similarity):")
        print("=" * 80)

        for res in results:
            print(f"\nRank {res['rank']} (Distance: {res['distance']:.4f})")
            print(f"Granularity: {res['metadata'].get('granularity', 'N/A')}")
            # Print relevant metadata
            if 'sales_total' in res['metadata']:
                print(f"Sales: ${res['metadata']['sales_total']:,.2f}")
            if 'profit_total' in res['metadata']:
                print(f"Profit: ${res['metadata']['profit_total']:,.2f}")
            if 'quantity_total' in res['metadata']:
                print(f"Quantity: {res['metadata']['quantity_total']:,}")
            if 'avg_discount' in res['metadata']:
                print(f"Avg Discount: {res['metadata']['avg_discount']:.2%}")

            print(f"Text: {res['text'][:150]}...")
            print("-" * 40)
        """
        Print retrieval results in a formatted way.

        Args:
            results: List of retrieval results
            query: Original query (optional)
        """
        if query:
            print(f"\n🔍 Query: {query}")

        print(f"📊 Top {len(results)} results (sorted by distance descending - worst to best similarity):")
        print("=" * 80)

        for res in results:
            print(f"\nRank {res['rank']} (Distance: {res['distance']:.4f})")
            print(f"Granularity: {res['metadata'].get('granularity', 'N/A')}")
            # Print relevant metadata
            if 'sales_total' in res['metadata']:
                print(f"Sales: ${res['metadata']['sales_total']:,.2f}")
            if 'profit_total' in res['metadata']:
                print(f"Profit: ${res['metadata']['profit_total']:,.2f}")
            if 'quantity_total' in res['metadata']:
                print(f"Quantity: {res['metadata']['quantity_total']:,}")
            if 'avg_discount' in res['metadata']:
                print(f"Avg Discount: {res['metadata']['avg_discount']:.2%}")

            print(f"Text: {res['text'][:150]}...")
            print("-" * 40)


# Example usage
if __name__ == "__main__":
    # Initialize retriever
    retriever = RAGRetriever()

    # Test queries
    test_queries = [
        "What are the sales in the South region?",
        "Which category has the highest profit?",
        "Show me discount impact on sales",
        "What is the total revenue?"
    ]

    print("\n" + "="*100)
    print("🧪 TESTING FAIRNESS-AWARE RERANKING")
    print("="*100)

    for query in test_queries:
        try:
            print(f"\n🔍 Testing Query: {query}")

            # Standard retrieval
            standard_results = retriever.retrieve(query, top_k=5)
            print(f"\n📊 Standard Retrieval (Top 5):")
            retriever.print_results(standard_results)

            # Fairness-aware retrieval
            fairness_results = retriever.retrieve_with_fairness(query, top_k=5)
            print(f"\n🎯 Fairness-Aware Retrieval (Top 5):")
            retriever.print_results(fairness_results)

            # Compare diversity
            print(f"\n📈 Diversity Comparison:")
            standard_regions = [r['metadata'].get('Region', 'Unknown') for r in standard_results]
            fairness_regions = [r['metadata'].get('Region', 'Unknown') for r in fairness_results]

            from collections import Counter
            print(f"Standard regions: {dict(Counter(standard_regions))}")
            print(f"Fairness regions: {dict(Counter(fairness_regions))}")

        except Exception as e:
            print(f"Error processing query '{query}': {e}")

    print("\n" + "="*100)
    print("🧪 TESTING PRE & POST FILTERING")
    print("="*100)

    # Test pre-filtering
    try:
        pre_filtered = retriever.pre_filter_by_metadata(
            "sales performance",
            {"Region": "South"},
            top_k=5
        )
        print(f"\n🔍 Pre-filtered results for South region:")
        retriever.print_results(pre_filtered, "sales performance (South only)")

    except Exception as e:
        print(f"Error in pre-filtering: {e}")

    # Test post-filtering
    try:
        all_results = retriever.retrieve("sales data", top_k=10)
        post_filtered = retriever.post_filter_diversity_check(all_results, max_region_ratio=0.5)
        print(f"\n🔍 Post-filtered results (max 50% same region):")
        retriever.print_results(post_filtered, "sales data (diversity filtered)")

    except Exception as e:
        print(f"Error in post-filtering: {e}")