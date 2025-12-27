from dataclasses import dataclass
import yaml

@dataclass
class AprilTagInfo:
    id: int
    size: float

@dataclass
class AprilTagConfig:
    family: str
    piano: AprilTagInfo
    hand: AprilTagInfo


def load_apriltag_config(path) -> AprilTagConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)["apriltags"]

    return AprilTagConfig(
        family=data["family"],
        piano=AprilTagInfo(
            id=data["piano"]["id"],
            size=data["piano"]["size"]
        ),
        hand=AprilTagInfo(
            id=data["hand"]["id"],
            size=data["hand"]["size"]
        )
    )
