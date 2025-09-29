"""Data transformation utilities for converting between backend and API models."""

import logging
from datetime import datetime
from typing import Any

from ...config.models.schemas import ConfigProfile, ConfigValue
from ...core.analysis.shot import ShotAnalysis

# Backend model imports
from ...core.models import (
    BallState,
    CueState,
    GameState,
    ShotType,
    TableState,
    Vector2D,
)
from ...core.physics.trajectory import Trajectory
from ...vision.models import Ball as VisionBall
from ...vision.models import CueStick as VisionCue
from ...vision.models import DetectionResult as VisionDetectionResult
from ...vision.models import Table as VisionTable
from .config_models import (
    ConfigProfileModel,
    ConfigSourceEnum,
    ConfigTypeEnum,
    ConfigurationModel,
    ConfigValueModel,
)

# API model imports
from .game_models import (
    BallModel,
    CueModel,
    GameStateModel,
    GameTypeEnum,
    PocketModel,
    ShotAnalysisModel,
    ShotTypeEnum,
    TableModel,
    TrajectoryModel,
    TrajectoryPointModel,
    Vector2DModel,
)
from .health_models import (
    HealthStatusEnum,
    IntegrationMetricsModel,
    PerformanceMetricsModel,
    ServiceHealthModel,
    ServiceTypeEnum,
)
from .vision_models import (
    BallTypeEnum,
    BoundingBoxModel,
    CueStateEnum,
    DetectionResultModel,
    Point2DModel,
    VisionBallModel,
    VisionCueModel,
    VisionTableModel,
)

logger = logging.getLogger(__name__)


class TransformationError(Exception):
    """Base exception for data transformation errors."""

    pass


class GameStateTransformer:
    """Transformer for game state related models."""

    @staticmethod
    def vector2d_to_model(vector: Vector2D) -> Vector2DModel:
        """Convert backend Vector2D to API model."""
        try:
            return Vector2DModel(x=vector.x, y=vector.y)
        except Exception as e:
            logger.error(f"Error converting Vector2D to model: {e}")
            raise TransformationError(f"Vector2D conversion failed: {e}")

    @staticmethod
    def model_to_vector2d(model: Vector2DModel) -> Vector2D:
        """Convert API Vector2DModel to backend Vector2D."""
        try:
            return Vector2D(x=model.x, y=model.y)
        except Exception as e:
            logger.error(f"Error converting Vector2DModel to backend: {e}")
            raise TransformationError(f"Vector2DModel conversion failed: {e}")

    @staticmethod
    def ball_state_to_model(ball: BallState) -> BallModel:
        """Convert backend BallState to API model."""
        try:
            return BallModel(
                id=ball.id,
                number=ball.number,
                position=GameStateTransformer.vector2d_to_model(ball.position),
                velocity=GameStateTransformer.vector2d_to_model(ball.velocity),
                radius=ball.radius,
                mass=ball.mass,
                is_cue_ball=ball.is_cue_ball,
                is_pocketed=ball.is_pocketed,
                color=getattr(ball, "color", None),
                confidence=getattr(ball, "confidence", 1.0),
            )
        except Exception as e:
            logger.error(f"Error converting BallState to model: {e}")
            raise TransformationError(f"BallState conversion failed: {e}")

    @staticmethod
    def model_to_ball_state(model: BallModel) -> BallState:
        """Convert API BallModel to backend BallState."""
        try:
            return BallState(
                id=model.id,
                position=GameStateTransformer.model_to_vector2d(model.position),
                velocity=GameStateTransformer.model_to_vector2d(model.velocity),
                radius=model.radius,
                mass=model.mass,
                spin=Vector2D(0, 0),  # Default spin
                is_cue_ball=model.is_cue_ball,
                is_pocketed=model.is_pocketed,
                number=model.number,
            )
        except Exception as e:
            logger.error(f"Error converting BallModel to backend: {e}")
            raise TransformationError(f"BallModel conversion failed: {e}")

    @staticmethod
    def table_state_to_model(table: TableState) -> TableModel:
        """Convert backend TableState to API model."""
        try:
            # Convert corners
            corners = [
                GameStateTransformer.vector2d_to_model(corner)
                for corner in table.corners
            ]

            # Convert pockets
            pockets = []
            if hasattr(table, "pockets") and table.pockets:
                for i, pocket_pos in enumerate(table.pockets):
                    pockets.append(
                        PocketModel(
                            id=f"pocket_{i}",
                            position=GameStateTransformer.vector2d_to_model(pocket_pos),
                            radius=0.06,  # Standard pocket radius
                            type="corner" if i < 4 else "side",
                        )
                    )

            return TableModel(
                width=table.width,
                height=table.height,
                corners=corners,
                pockets=pockets,
                surface_color=getattr(table, "surface_color", None),
            )
        except Exception as e:
            logger.error(f"Error converting TableState to model: {e}")
            raise TransformationError(f"TableState conversion failed: {e}")

    @staticmethod
    def cue_state_to_model(cue: CueState) -> CueModel:
        """Convert backend CueState to API model."""
        try:
            return CueModel(
                tip_position=GameStateTransformer.vector2d_to_model(cue.tip_position),
                angle=cue.angle,
                length=getattr(cue, "length", 1.47),
                is_aiming=getattr(cue, "is_aiming", False),
                confidence=getattr(cue, "confidence", 1.0),
            )
        except Exception as e:
            logger.error(f"Error converting CueState to model: {e}")
            raise TransformationError(f"CueState conversion failed: {e}")

    @staticmethod
    def game_state_to_model(state: GameState) -> GameStateModel:
        """Convert backend GameState to API model."""
        try:
            # Convert balls
            balls = [
                GameStateTransformer.ball_state_to_model(ball) for ball in state.balls
            ]

            # Convert table
            table = GameStateTransformer.table_state_to_model(state.table)

            # Convert cue if present
            cue = None
            if state.cue:
                cue = GameStateTransformer.cue_state_to_model(state.cue)

            # Convert game type
            game_type = GameTypeEnum.PRACTICE
            if hasattr(state, "game_type"):
                game_type_map = {
                    "practice": GameTypeEnum.PRACTICE,
                    "8-ball": GameTypeEnum.EIGHT_BALL,
                    "9-ball": GameTypeEnum.NINE_BALL,
                    "straight_pool": GameTypeEnum.STRAIGHT_POOL,
                }
                game_type = game_type_map.get(
                    str(state.game_type).lower(), GameTypeEnum.PRACTICE
                )

            return GameStateModel(
                timestamp=state.timestamp,
                frame_number=getattr(state, "frame_number", None),
                game_type=game_type,
                balls=balls,
                table=table,
                cue=cue,
                current_player=getattr(state, "current_player", None),
                turn_number=getattr(state, "turn_number", 1),
                is_game_active=getattr(state, "is_game_active", True),
                metadata=getattr(state, "metadata", {}),
            )
        except Exception as e:
            logger.error(f"Error converting GameState to model: {e}")
            raise TransformationError(f"GameState conversion failed: {e}")

    @staticmethod
    def shot_analysis_to_model(analysis: ShotAnalysis) -> ShotAnalysisModel:
        """Convert backend ShotAnalysis to API model."""
        try:
            # Convert shot type
            shot_type = ShotTypeEnum.STRAIGHT
            if hasattr(analysis, "shot_type"):
                shot_type_map = {
                    ShotType.STRAIGHT: ShotTypeEnum.STRAIGHT,
                    ShotType.BANK: ShotTypeEnum.BANK,
                    ShotType.COMBINATION: ShotTypeEnum.COMBINATION,
                    ShotType.CAROM: ShotTypeEnum.CAROM,
                    ShotType.BREAK: ShotTypeEnum.BREAK,
                    ShotType.SAFETY: ShotTypeEnum.SAFETY,
                }
                shot_type = shot_type_map.get(analysis.shot_type, ShotTypeEnum.STRAIGHT)

            # Convert trajectory if present
            predicted_trajectory = None
            if (
                hasattr(analysis, "predicted_trajectory")
                and analysis.predicted_trajectory
            ):
                predicted_trajectory = GameStateTransformer.trajectory_to_model(
                    analysis.predicted_trajectory
                )

            # Convert alternative shots
            alternatives = []
            if hasattr(analysis, "alternative_shots") and analysis.alternative_shots:
                alternatives = [
                    GameStateTransformer.shot_analysis_to_model(alt)
                    for alt in analysis.alternative_shots[:5]  # Limit alternatives
                ]

            return ShotAnalysisModel(
                shot_type=shot_type,
                target_ball=getattr(analysis, "target_ball", None),
                recommended_angle=getattr(analysis, "recommended_angle", 0.0),
                recommended_force=getattr(analysis, "recommended_force", 0.5),
                success_probability=getattr(analysis, "success_probability", 0.0),
                difficulty=getattr(analysis, "difficulty", 0.5),
                predicted_trajectory=predicted_trajectory,
                alternative_shots=alternatives,
                warnings=getattr(analysis, "warnings", []),
                tips=getattr(analysis, "tips", []),
            )
        except Exception as e:
            logger.error(f"Error converting ShotAnalysis to model: {e}")
            raise TransformationError(f"ShotAnalysis conversion failed: {e}")

    @staticmethod
    def trajectory_to_model(trajectory: Trajectory) -> TrajectoryModel:
        """Convert backend Trajectory to API model."""
        try:
            # Convert trajectory points
            points = []
            if hasattr(trajectory, "points") and trajectory.points:
                for point in trajectory.points:
                    points.append(
                        TrajectoryPointModel(
                            position=GameStateTransformer.vector2d_to_model(
                                point.position
                            ),
                            time=point.time,
                            velocity=GameStateTransformer.vector2d_to_model(
                                point.velocity
                            ),
                        )
                    )

            # Get end position
            end_position = points[-1].position if points else Vector2DModel(x=0, y=0)

            return TrajectoryModel(
                ball_id=getattr(trajectory, "ball_id", ""),
                points=points,
                total_time=getattr(trajectory, "total_time", 0.0),
                end_position=end_position,
                collisions=getattr(trajectory, "collisions", []),
            )
        except Exception as e:
            logger.error(f"Error converting Trajectory to model: {e}")
            raise TransformationError(f"Trajectory conversion failed: {e}")


class VisionTransformer:
    """Transformer for vision related models."""

    @staticmethod
    def vision_ball_to_model(ball: VisionBall) -> VisionBallModel:
        """Convert backend vision Ball to API model."""
        try:
            # Convert ball type
            ball_type = BallTypeEnum.UNKNOWN
            if hasattr(ball, "ball_type"):
                type_map = {
                    "cue": BallTypeEnum.CUE,
                    "solid": BallTypeEnum.SOLID,
                    "stripe": BallTypeEnum.STRIPE,
                    "eight": BallTypeEnum.EIGHT,
                }
                ball_type = type_map.get(
                    str(ball.ball_type).lower(), BallTypeEnum.UNKNOWN
                )

            return VisionBallModel(
                id=ball.id,
                center=Point2DModel(x=ball.center.x, y=ball.center.y),
                radius=ball.radius,
                color=tuple(ball.color) if hasattr(ball, "color") else (255, 255, 255),
                ball_type=ball_type,
                number=getattr(ball, "number", None),
                confidence=getattr(ball, "confidence", 1.0),
                bounding_box=BoundingBoxModel(
                    x=ball.center.x - ball.radius,
                    y=ball.center.y - ball.radius,
                    width=ball.radius * 2,
                    height=ball.radius * 2,
                    confidence=getattr(ball, "confidence", 1.0),
                ),
                is_moving=getattr(ball, "is_moving", False),
                velocity_pixels=(
                    Point2DModel(x=0, y=0)
                    if not hasattr(ball, "velocity")
                    else Point2DModel(x=ball.velocity.x, y=ball.velocity.y)
                ),
            )
        except Exception as e:
            logger.error(f"Error converting vision Ball to model: {e}")
            raise TransformationError(f"Vision Ball conversion failed: {e}")

    @staticmethod
    def vision_table_to_model(table: VisionTable) -> VisionTableModel:
        """Convert backend vision Table to API model."""
        try:
            # Convert corners
            corners = [
                Point2DModel(x=corner[0], y=corner[1]) for corner in table.corners
            ]

            # Convert pockets
            pockets = []
            if hasattr(table, "pockets") and table.pockets:
                for i, pocket in enumerate(table.pockets):
                    pockets.append(
                        PocketModel(
                            id=f"pocket_{i}",
                            center=Point2DModel(x=pocket[0], y=pocket[1]),
                            radius=30.0,  # Default pocket radius in pixels
                            confidence=1.0,
                        )
                    )

            return VisionTableModel(
                corners=corners,
                pockets=pockets,
                surface_color=(
                    tuple(table.surface_color)
                    if hasattr(table, "surface_color")
                    else (60, 200, 100)
                ),
                width_pixels=table.width,
                height_pixels=table.height,
                confidence=getattr(table, "confidence", 1.0),
            )
        except Exception as e:
            logger.error(f"Error converting vision Table to model: {e}")
            raise TransformationError(f"Vision Table conversion failed: {e}")

    @staticmethod
    def vision_cue_to_model(cue: VisionCue) -> VisionCueModel:
        """Convert backend vision CueStick to API model."""
        try:
            cue_state = CueStateEnum.DETECTED
            if hasattr(cue, "state"):
                state_map = {
                    "not_detected": CueStateEnum.NOT_DETECTED,
                    "detected": CueStateEnum.DETECTED,
                    "aiming": CueStateEnum.AIMING,
                    "striking": CueStateEnum.STRIKING,
                }
                cue_state = state_map.get(str(cue.state).lower(), CueStateEnum.DETECTED)

            # Calculate end position based on tip and angle
            end_x = cue.tip_position.x - cue.length * 50  # Approximate pixels per meter
            end_y = cue.tip_position.y - cue.length * 50

            return VisionCueModel(
                tip_position=Point2DModel(x=cue.tip_position.x, y=cue.tip_position.y),
                end_position=Point2DModel(x=end_x, y=end_y),
                angle=cue.angle,
                length_pixels=cue.length * 50,  # Convert to pixels
                state=cue_state,
                confidence=getattr(cue, "confidence", 1.0),
                bounding_box=BoundingBoxModel(
                    x=min(cue.tip_position.x, end_x) - 10,
                    y=min(cue.tip_position.y, end_y) - 10,
                    width=abs(cue.tip_position.x - end_x) + 20,
                    height=abs(cue.tip_position.y - end_y) + 20,
                    confidence=getattr(cue, "confidence", 1.0),
                ),
                is_visible=getattr(cue, "is_visible", True),
            )
        except Exception as e:
            logger.error(f"Error converting vision CueStick to model: {e}")
            raise TransformationError(f"Vision CueStick conversion failed: {e}")

    @staticmethod
    def detection_result_to_model(
        result: VisionDetectionResult,
    ) -> DetectionResultModel:
        """Convert backend DetectionResult to API model."""
        try:
            # Convert balls
            balls = []
            if result.balls:
                balls = [
                    VisionTransformer.vision_ball_to_model(ball)
                    for ball in result.balls
                ]

            # Convert table
            table = None
            if result.table:
                table = VisionTransformer.vision_table_to_model(result.table)

            # Convert cue
            cue = None
            if result.cue:
                cue = VisionTransformer.vision_cue_to_model(result.cue)

            return DetectionResultModel(
                frame_number=result.frame_number,
                timestamp=result.timestamp,
                processing_time=result.processing_time,
                balls=balls,
                table=table,
                cue=cue,
                frame_width=getattr(result, "frame_width", 1920),
                frame_height=getattr(result, "frame_height", 1080),
                detection_quality=getattr(result, "detection_quality", {}),
                metadata=getattr(result, "metadata", {}),
            )
        except Exception as e:
            logger.error(f"Error converting DetectionResult to model: {e}")
            raise TransformationError(f"DetectionResult conversion failed: {e}")


class ConfigurationTransformer:
    """Transformer for configuration related models."""

    @staticmethod
    def config_value_to_model(config_value: ConfigValue) -> ConfigValueModel:
        """Convert backend ConfigValue to API model."""
        try:
            # Determine config type
            config_type = ConfigTypeEnum.STRING
            if isinstance(config_value.value, bool):
                config_type = ConfigTypeEnum.BOOLEAN
            elif isinstance(config_value.value, int):
                config_type = ConfigTypeEnum.INTEGER
            elif isinstance(config_value.value, float):
                config_type = ConfigTypeEnum.FLOAT
            elif isinstance(config_value.value, list):
                config_type = ConfigTypeEnum.LIST
            elif isinstance(config_value.value, dict):
                config_type = ConfigTypeEnum.DICT

            # Determine source
            source = ConfigSourceEnum.DEFAULT
            if hasattr(config_value, "source"):
                source_map = {
                    "default": ConfigSourceEnum.DEFAULT,
                    "file": ConfigSourceEnum.FILE,
                    "environment": ConfigSourceEnum.ENVIRONMENT,
                    "user": ConfigSourceEnum.USER,
                    "runtime": ConfigSourceEnum.RUNTIME,
                }
                source = source_map.get(
                    str(config_value.source).lower(), ConfigSourceEnum.DEFAULT
                )

            return ConfigValueModel(
                key=config_value.key,
                value=config_value.value,
                type=config_type,
                source=source,
                description=getattr(config_value, "description", None),
                default_value=getattr(config_value, "default", None),
                is_required=getattr(config_value, "required", False),
                is_secret=getattr(config_value, "secret", False),
                validation_rules=getattr(config_value, "validation", None),
                last_modified=getattr(config_value, "modified_at", None),
                modified_by=getattr(config_value, "modified_by", None),
            )
        except Exception as e:
            logger.error(f"Error converting ConfigValue to model: {e}")
            raise TransformationError(f"ConfigValue conversion failed: {e}")

    @staticmethod
    def config_profile_to_model(profile: ConfigProfile) -> ConfigProfileModel:
        """Convert backend ConfigProfile to API model."""
        try:
            # Convert configuration data
            configuration = ConfigurationModel(
                version=getattr(profile, "version", "1.0"),
                timestamp=getattr(profile, "created_at", datetime.now()),
                metadata=getattr(profile, "metadata", {}),
            )

            return ConfigProfileModel(
                name=profile.name,
                description=getattr(profile, "description", None),
                configuration=configuration,
                is_active=getattr(profile, "is_active", False),
                created_at=getattr(profile, "created_at", datetime.now()),
                updated_at=getattr(profile, "updated_at", None),
                tags=getattr(profile, "tags", []),
            )
        except Exception as e:
            logger.error(f"Error converting ConfigProfile to model: {e}")
            raise TransformationError(f"ConfigProfile conversion failed: {e}")


class HealthTransformer:
    """Transformer for health and monitoring related models."""

    @staticmethod
    def service_health_to_model(
        name: str, status: str, details: dict[str, Any]
    ) -> ServiceHealthModel:
        """Convert service health data to API model."""
        try:
            # Convert status
            health_status = HealthStatusEnum.UNKNOWN
            status_map = {
                "healthy": HealthStatusEnum.HEALTHY,
                "degraded": HealthStatusEnum.DEGRADED,
                "unhealthy": HealthStatusEnum.UNHEALTHY,
            }
            health_status = status_map.get(status.lower(), HealthStatusEnum.UNKNOWN)

            # Determine service type
            service_type = ServiceTypeEnum.EXTERNAL
            type_map = {
                "core": ServiceTypeEnum.CORE,
                "config": ServiceTypeEnum.CONFIGURATION,
                "configuration": ServiceTypeEnum.CONFIGURATION,
                "vision": ServiceTypeEnum.VISION,
                "api": ServiceTypeEnum.API,
                "database": ServiceTypeEnum.DATABASE,
                "cache": ServiceTypeEnum.CACHE,
            }
            service_type = type_map.get(name.lower(), ServiceTypeEnum.EXTERNAL)

            return ServiceHealthModel(
                name=name,
                type=service_type,
                status=health_status,
                last_check=datetime.now(),
                response_time=details.get("response_time"),
                error_message=details.get("error_message"),
                details=details,
                dependencies=details.get("dependencies", []),
                version=details.get("version"),
                uptime=details.get("uptime"),
            )
        except Exception as e:
            logger.error(f"Error converting service health to model: {e}")
            raise TransformationError(f"Service health conversion failed: {e}")

    @staticmethod
    def performance_metrics_to_model(
        metrics: dict[str, Any],
    ) -> PerformanceMetricsModel:
        """Convert performance metrics to API model."""
        try:
            return PerformanceMetricsModel(
                requests_total=metrics.get("requests_total", 0),
                requests_per_second=metrics.get("requests_per_second", 0.0),
                response_time_avg=metrics.get("response_time_avg", 0.0),
                response_time_p50=metrics.get("response_time_p50", 0.0),
                response_time_p95=metrics.get("response_time_p95", 0.0),
                response_time_p99=metrics.get("response_time_p99", 0.0),
                error_rate=metrics.get("error_rate", 0.0),
                cache_hit_rate=metrics.get("cache_hit_rate", 0.0),
                active_connections=metrics.get("active_connections", 0),
                queue_size=metrics.get("queue_size", 0),
            )
        except Exception as e:
            logger.error(f"Error converting performance metrics to model: {e}")
            raise TransformationError(f"Performance metrics conversion failed: {e}")

    @staticmethod
    def integration_metrics_to_model(
        metrics: dict[str, Any],
    ) -> IntegrationMetricsModel:
        """Convert integration metrics to API model."""
        try:
            return IntegrationMetricsModel(
                modules_initialized=metrics.get("modules_initialized", 0),
                services_healthy=metrics.get("services_healthy", 0),
                total_services=metrics.get("total_services", 0),
                events_processed=metrics.get("events_processed", 0),
                cache_entries=metrics.get("cache_entries", 0),
                background_tasks_active=metrics.get("background_tasks_active", 0),
                integration_uptime=metrics.get("integration_uptime", 0.0),
                last_health_check=metrics.get("last_health_check"),
            )
        except Exception as e:
            logger.error(f"Error converting integration metrics to model: {e}")
            raise TransformationError(f"Integration metrics conversion failed: {e}")


# Utility functions for batch transformations


def transform_dict_to_api_models(
    data: dict[str, Any], transformer_type: str
) -> dict[str, Any]:
    """Transform a dictionary of backend data to API models."""
    try:
        transformed = {}

        if (
            transformer_type == "game"
            or transformer_type == "vision"
            or transformer_type == "config"
            or transformer_type == "health"
        ):
            pass
        else:
            raise ValueError(f"Unknown transformer type: {transformer_type}")

        # Apply appropriate transformations based on data structure
        for key, value in data.items():
            if isinstance(value, dict):
                transformed[key] = transform_dict_to_api_models(value, transformer_type)
            elif isinstance(value, list):
                transformed[key] = [
                    (
                        transform_dict_to_api_models(item, transformer_type)
                        if isinstance(item, dict)
                        else item
                    )
                    for item in value
                ]
            else:
                transformed[key] = value

        return transformed

    except Exception as e:
        logger.error(f"Error in batch transformation: {e}")
        raise TransformationError(f"Batch transformation failed: {e}")


def safe_transform(transform_func, data, default=None, log_errors=True):
    """Safely apply a transformation function with error handling."""
    try:
        if data is None:
            return default
        return transform_func(data)
    except Exception as e:
        if log_errors:
            logger.warning(f"Transformation failed: {e}")
        return default


__all__ = [
    "TransformationError",
    "GameStateTransformer",
    "VisionTransformer",
    "ConfigurationTransformer",
    "HealthTransformer",
    "transform_dict_to_api_models",
    "safe_transform",
]
