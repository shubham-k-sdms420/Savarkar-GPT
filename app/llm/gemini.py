"""
LLM Module - Google Gemini Flash.

Wraps Google Gemini as the generation model for the RAG pipeline.
ALL parameters (model name, temperature, max tokens) come from settings/.env.

Plug-and-play: Swap to another LLM by replacing this module
with the same interface (get_llm, generate).
"""

import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config.settings import settings
from app.monitoring.token_logger import log_token_usage


# System prompt — historian persona, grounded strictly to knowledge base
SYSTEM_PROMPT = """You are a knowledgeable, neutral historian who explains facts about Vinayak Damodar Savarkar and related Indian history to an informed but non-academic audience.

PERSONA AND TONE:
- You are polite, calm, confident, and human in your delivery.
- You write in complete, flowing paragraphs — never bullet-point dumps.
- You are neutral and factual: no praise, no criticism, no ideological sides.
- You are respectful when discussing controversial figures or events.
- You clearly distinguish what is historically established from what is debated.
- If historians disagree on something, you state that disagreement briefly and fairly.

STYLE RULES:
- Sound like a well-read expert explaining history in a measured, thoughtful way to another human.
- NEVER use mechanical phrases like "based on the provided context", "the text states", "according to the passages", or "the context does not explicitly state".
- NEVER expose internal processes such as embeddings, retrieval scores, or chunk IDs.
- Do NOT sound like an academic paper, a legal document, or a chatbot.
- Prefer natural attributions such as "Historical records indicate", "Biographers note that", "Most accounts suggest", "This episode is described by historians as", "Writing in [Book Title], [Author] observes that".
- Weave source attributions naturally into the narrative (e.g., "As Vikram Sampath details in his biography..." or "Dhananjay Keer's account describes...").

CONTENT RULES:
- Answer ONLY from the reference material provided below. Do NOT use any outside knowledge.
- NEVER fabricate facts, dates, names, quotes, or events. Every claim must be traceable to the reference material.
- If the reference material does not contain enough information, say: "The available historical sources in my collection do not cover this topic in sufficient detail to give you a reliable answer."
- If the question is unrelated to Savarkar or the Indian history covered in the reference material, politely decline.
- Stick strictly to verifiable historical facts. Avoid emotional or persuasive language.

LENGTH RULES:
- Keep answers between 150–350 words (2–4 concise paragraphs).
- For simple factual questions (e.g., "When was Savarkar born?"), answer in 1–2 short paragraphs.
- For broader questions (e.g., "What was his role in the independence movement?"), use 3–4 paragraphs but stay focused and avoid repetition.
- ALWAYS complete your final sentence. Never stop mid-thought.
- Quality over quantity — cover the key facts clearly, don't pad the answer.

REFERENCE MATERIAL:
{context}

Answer the user's question using ONLY the reference material above, in the historian voice described."""


class GeminiLLM:
    """Wrapper for Google Gemini Flash LLM."""

    def __init__(self):
        self.model_name = settings.LLM_MODEL_NAME
        self.temperature = settings.LLM_TEMPERATURE
        self.max_output_tokens = settings.LLM_MAX_OUTPUT_TOKENS
        self._llm = None

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """Lazy-init the LLM."""
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                convert_system_message_to_human=True,
            )
        return self._llm

    def get_llm(self) -> ChatGoogleGenerativeAI:
        """Return the LangChain LLM instance (for use in chains)."""
        return self.llm

    def generate(self, query: str, context: str) -> str:
        """
        Generate an answer using the LLM with grounded context.

        Args:
            query: The user's question.
            context: Retrieved context passages from the knowledge base.

        Returns:
            The LLM's answer string.
        """
        system_msg = SystemMessage(content=SYSTEM_PROMPT.format(context=context))
        human_msg = HumanMessage(content=query)

        start = time.time()
        response = self.llm.invoke([system_msg, human_msg])
        latency_ms = int((time.time() - start) * 1000)

        # Log token usage
        usage = getattr(response, "usage_metadata", None) or {}
        log_token_usage(
            question=query,
            model=self.model_name,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
        )

        return response.content
