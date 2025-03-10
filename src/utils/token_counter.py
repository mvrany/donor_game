from threading import Lock
from dataclasses import dataclass
from typing import Dict

@dataclass
class TokenCount:
    """Store token counts for input and output."""
    input_tokens: int = 0
    output_tokens: int = 0

class TokenCounter:
    """Thread-safe token counter for tracking LLM usage."""
    
    def __init__(self):
        self._lock = Lock()
        self._counts: Dict[str, TokenCount] = {}
        self._total = TokenCount()
    
    def add_tokens(self, thread_id: str, input_tokens: int, output_tokens: int):
        """Add token counts for a specific thread in a thread-safe manner."""
        with self._lock:
            if thread_id not in self._counts:
                self._counts[thread_id] = TokenCount()
            
            self._counts[thread_id].input_tokens += input_tokens
            self._counts[thread_id].output_tokens += output_tokens
            
            self._total.input_tokens += input_tokens
            self._total.output_tokens += output_tokens
    
    def get_thread_counts(self) -> Dict[str, TokenCount]:
        """Get token counts per thread."""
        with self._lock:
            return self._counts.copy()
    
    def get_total_counts(self) -> TokenCount:
        """Get total token counts across all threads."""
        with self._lock:
            return TokenCount(
                input_tokens=self._total.input_tokens,
                output_tokens=self._total.output_tokens
            )
    
    def reset(self):
        """Reset all counts."""
        with self._lock:
            self._counts.clear()
            self._total = TokenCount()

# Global token counter instance
token_counter = TokenCounter() 