import os
from typing import List, Dict, Any
from retriever import RAGRetriever

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("   Or set environment variables manually")

class RAGWithLLM:
    """
    Complete RAG system with Google Gemini LLM integration for e-commerce analytics.
    """

    def __init__(self, retriever: RAGRetriever, model_name: str = "gemini-2.5-flash"):
        """
        Initialize RAG with Google Gemini.

        Args:
            retriever: RAGRetriever instance
            model_name: Google Gemini model name (e.g., "gemini-2.5-flash", "gemini-pro-latest")
        """
        self.retriever = retriever
        self.model_name = model_name

        # Initialize Google Gemini client
        self._init_llm_client()

    def _init_llm_client(self):
        """Initialize Google Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            self.client = genai.GenerativeModel(self.model_name)
            print(f"✅ Google Gemini initialized: {self.model_name}")
        except ImportError as e:
            raise ImportError(f"Missing required package for Google Gemini: {e}")
        except Exception as e:
            raise Exception(f"Error initializing Google Gemini client: {e}")

    def _format_context(self, retrieval_results: List[Dict]) -> str:
        """Format retrieved documents into context for LLM."""
        context_parts = []

        for i, result in enumerate(retrieval_results, 1):
            meta = result['metadata']
            text = result['text']

            # Format metadata info
            info_parts = []
            if 'granularity' in meta:
                info_parts.append(f"Level: {meta['granularity']}")
            if 'Region' in meta:
                info_parts.append(f"Region: {meta['Region']}")
            if 'Category' in meta:
                info_parts.append(f"Category: {meta['Category']}")
            if 'sales_total' in meta:
                info_parts.append(f"Sales: ${meta['sales_total']:,.2f}")
            if 'profit_total' in meta:
                info_parts.append(f"Profit: ${meta['profit_total']:,.2f}")

            info_str = " | ".join(info_parts)
            context_parts.append(f"[{i}] {info_str}\n{text}")

        return "\n\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        """Create the prompt for Google Gemini."""
        system_prompt = """You are an expert e-commerce analyst. Use the provided sales data context to answer questions accurately and comprehensively.

Guidelines:
- Base your answer on the provided data only
- Be specific with numbers and percentages
- Explain trends and insights clearly
- If data is insufficient, say so clearly
- Structure your answer with sections if appropriate
- Use markdown formatting for readability

Context contains sales data at different granularities (overall, region, category, etc.)."""

        user_prompt = f"""Question: {query}

Context Data:
{context}

Please provide a comprehensive analysis based on the above data."""

        return f"{system_prompt}\n\n{user_prompt}"

    def query(self, user_query: str, top_k: int = 5, use_fairness: bool = True) -> Dict[str, Any]:
        """
        Complete RAG query: retrieve + generate answer.

        Args:
            user_query: User's question
            top_k: Number of documents to retrieve
            use_fairness: Whether to use fairness-aware retrieval

        Returns:
            Dict with answer, retrieved_docs, and metadata
        """
        try:
            # Step 1: Retrieve relevant documents
            if use_fairness:
                retrieval_results = self.retriever.retrieve_with_fairness(
                    user_query, top_k=top_k, candidate_pool_size=max(20, top_k * 3)
                )
            else:
                retrieval_results = self.retriever.retrieve(user_query, top_k=top_k)

            # Step 2: Format context
            context = self._format_context(retrieval_results)

            # Step 3: Create prompt
            prompt = self._create_prompt(user_query, context)

            # Step 4: Generate answer
            answer = self._generate_answer(prompt)

            return {
                "answer": answer,
                "retrieved_docs": retrieval_results,
                "query": user_query,
                "num_docs": len(retrieval_results),
                "llm_provider": "google",
                "model": self.model_name
            }

        except Exception as e:
            raise Exception(f"Error in RAG query: {e}")

    def _generate_answer(self, prompt: str) -> str:
        """Generate answer using Google Gemini."""
        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Error generating answer with google: {e}")