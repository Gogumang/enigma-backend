class DomainException(Exception):
    """Base domain exception"""
    def __init__(self, message: str, code: str = "DOMAIN_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


class ValidationException(DomainException):
    """Validation error"""
    def __init__(self, message: str):
        super().__init__(message, "VALIDATION_ERROR")


class ExternalServiceException(DomainException):
    """External service error"""
    def __init__(self, message: str, service: str):
        super().__init__(f"{service}: {message}", "EXTERNAL_SERVICE_ERROR")


