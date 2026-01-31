import uuid
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ScammerEntity:
    """스캐머 도메인 엔티티"""
    id: str
    name: str
    face_embedding: list[float]
    report_count: int = 1
    source: str | None = None
    reported_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(
        cls,
        name: str,
        face_embedding: list[float],
        source: str | None = None
    ) -> "ScammerEntity":
        """새 스캐머 엔티티 생성"""
        return cls(
            id=f"scammer_{uuid.uuid4().hex[:12]}",
            name=name,
            face_embedding=face_embedding,
            source=source
        )

    def increment_report(self) -> None:
        """신고 횟수 증가"""
        self.report_count += 1
        self.updated_at = datetime.now()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "face_embedding": self.face_embedding,
            "report_count": self.report_count,
            "source": self.source,
            "reported_at": self.reported_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScammerEntity":
        return cls(
            id=data["id"],
            name=data["name"],
            face_embedding=data["face_embedding"],
            report_count=data.get("report_count", 1),
            source=data.get("source"),
            reported_at=datetime.fromisoformat(data["reported_at"]) if isinstance(data.get("reported_at"), str) else data.get("reported_at", datetime.now()),
            updated_at=datetime.fromisoformat(data["updated_at"]) if isinstance(data.get("updated_at"), str) else data.get("updated_at", datetime.now())
        )
