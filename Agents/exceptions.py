class AgentError(Exception):
    """Base exception for agent errors."""
    pass

class AgentInitializationError(AgentError):
    """Raised when agent initialization fails."""
    pass

class SearchOperationError(AgentError):
    """Raised when search operation fails."""
    pass

class DatabaseConnectionError(AgentError):
    """Raised when database connection fails."""
    pass

class PredictionError(AgentError):
    """Raised when model prediction or condition identification fails."""
    pass


class CommunicationError(AgentError):
    """Raised when text generation or explanation fails."""
    pass