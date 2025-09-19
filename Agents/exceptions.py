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