from .chat_router import router as chat_router
from .comprehensive_router import router as comprehensive_router
from .deepfake_router import router as deepfake_router
from .dependencies import initialize_services
from .fraud_router import router as fraud_router
from .profile_router import router as profile_router
from .report_router import router as report_router
from .training_router import router as training_router
from .url_router import router as url_router
from .verify_router import router as verify_router

__all__ = [
    "profile_router",
    "deepfake_router",
    "chat_router",
    "comprehensive_router",
    "fraud_router",
    "report_router",
    "training_router",
    "url_router",
    "verify_router",
    "initialize_services"
]
