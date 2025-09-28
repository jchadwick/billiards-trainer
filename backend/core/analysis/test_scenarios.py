"""Comprehensive test scenarios for shot analysis and assistance."""

from dataclasses import dataclass
from typing import Any

from backend.core.models import (
    BallState,
    GameState,
    GameType,
    ShotType,
    TableState,
    Vector2D,
)

from .assistance import AssistanceEngine, AssistanceLevel
from .prediction import OutcomePredictor
from .shot import ShotAnalyzer


@dataclass
class TestScenario:
    """A test scenario for analysis components."""

    name: str
    description: str
    game_state: GameState
    expected_shot_type: ShotType
    expected_difficulty_range: tuple  # (min, max)
    expected_success_range: tuple  # (min, max)
    expected_problems: list[str]
    test_assertions: list[str]  # Human-readable test expectations


class AnalysisTestScenarios:
    """Comprehensive test scenarios for analysis accuracy."""

    def __init__(self):
        self.shot_analyzer = ShotAnalyzer()
        self.outcome_predictor = OutcomePredictor()
        self.assistance_engine = AssistanceEngine()

    def get_all_scenarios(self) -> list[TestScenario]:
        """Get all test scenarios."""
        scenarios = []

        # Basic shot scenarios
        scenarios.extend(self._create_basic_shot_scenarios())

        # Difficulty scenarios
        scenarios.extend(self._create_difficulty_scenarios())

        # Game type scenarios
        scenarios.extend(self._create_game_type_scenarios())

        # Problem scenarios
        scenarios.extend(self._create_problem_scenarios())

        # Advanced scenarios
        scenarios.extend(self._create_advanced_scenarios())

        return scenarios

    def _create_basic_shot_scenarios(self) -> list[TestScenario]:
        """Create basic shot type scenarios."""
        scenarios = []

        # Direct shot scenario
        table = TableState.standard_9ft_table()
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.5, 0.6), number=1),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=table,
            game_type=GameType.PRACTICE,
        )

        scenarios.append(
            TestScenario(
                name="direct_shot_easy",
                description="Simple direct shot with clear path to pocket",
                game_state=game_state,
                expected_shot_type=ShotType.DIRECT,
                expected_difficulty_range=(0.0, 0.4),
                expected_success_range=(0.6, 1.0),
                expected_problems=[],
                test_assertions=[
                    "Shot type should be DIRECT",
                    "Difficulty should be low (easy shot)",
                    "Success probability should be high",
                    "No major problems should be identified",
                ],
            )
        )

        # Bank shot scenario
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.3), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(0.5, 0.9), number=1),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=table,
            game_type=GameType.PRACTICE,
        )

        scenarios.append(
            TestScenario(
                name="bank_shot_moderate",
                description="Bank shot off cushion required",
                game_state=game_state,
                expected_shot_type=ShotType.BANK,
                expected_difficulty_range=(0.5, 0.8),
                expected_success_range=(0.3, 0.7),
                expected_problems=[],
                test_assertions=[
                    "Shot type should be BANK",
                    "Difficulty should be moderate to high",
                    "Success probability should be moderate",
                    "Should identify cushion contact requirement",
                ],
            )
        )

        return scenarios

    def _create_difficulty_scenarios(self) -> list[TestScenario]:
        """Create scenarios testing difficulty assessment."""
        scenarios = []
        table = TableState.standard_9ft_table()

        # Very easy shot
        balls = [
            BallState(id="cue", position=Vector2D(0.8, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.0, 0.6), number=1),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=table,
            game_type=GameType.PRACTICE,
        )

        scenarios.append(
            TestScenario(
                name="very_easy_shot",
                description="Very short straight shot",
                game_state=game_state,
                expected_shot_type=ShotType.DIRECT,
                expected_difficulty_range=(0.0, 0.2),
                expected_success_range=(0.8, 1.0),
                expected_problems=[],
                test_assertions=[
                    "Should be rated as very easy",
                    "Success probability should be very high",
                    "Recommended force should be low",
                ],
            )
        )

        # Very difficult shot
        balls = [
            BallState(id="cue", position=Vector2D(0.2, 0.2), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(2.3, 1.0), number=1),
            # Add interfering balls
            BallState(id="ball_2", position=Vector2D(1.0, 0.4), number=2),
            BallState(id="ball_3", position=Vector2D(1.5, 0.7), number=3),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=table,
            game_type=GameType.PRACTICE,
        )

        scenarios.append(
            TestScenario(
                name="very_difficult_shot",
                description="Long shot with obstacles and thin cut",
                game_state=game_state,
                expected_shot_type=ShotType.COMBINATION,
                expected_difficulty_range=(0.7, 1.0),
                expected_success_range=(0.0, 0.4),
                expected_problems=["Direct path blocked", "Long distance shot"],
                test_assertions=[
                    "Should be rated as very difficult",
                    "Success probability should be low",
                    "Should identify path obstacles",
                    "Should suggest alternative approaches",
                ],
            )
        )

        return scenarios

    def _create_game_type_scenarios(self) -> list[TestScenario]:
        """Create scenarios for different game types."""
        scenarios = []
        table = TableState.standard_9ft_table()

        # 9-ball scenario - must hit lowest number first
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.0, 0.6), number=1),
            BallState(id="ball_2", position=Vector2D(1.5, 0.6), number=2),
            BallState(id="ball_3", position=Vector2D(2.0, 0.6), number=3),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=table,
            game_type=GameType.NINE_BALL,
        )

        scenarios.append(
            TestScenario(
                name="nine_ball_legal_target",
                description="9-ball game - should target lowest numbered ball",
                game_state=game_state,
                expected_shot_type=ShotType.DIRECT,
                expected_difficulty_range=(0.0, 0.4),
                expected_success_range=(0.6, 1.0),
                expected_problems=[],
                test_assertions=[
                    "Should identify ball 1 as primary target",
                    "Should not suggest ball 2 or 3 as primary targets",
                    "Shot should be legal",
                    "Strategic value should be high for ball 1",
                ],
            )
        )

        # Break shot scenario
        # Create standard rack formation
        balls = [BallState(id="cue", position=Vector2D(0.6, 0.6), is_cue_ball=True)]

        # Add racked balls in triangle formation
        rack_center = Vector2D(1.9, 0.6)
        ball_diameter = 0.057
        rack_positions = [
            (0, 0),  # 1 ball (front)
            (-1, -0.5),
            (-1, 0.5),  # 2nd row
            (-2, -1),
            (-2, 0),
            (-2, 1),  # 3rd row
            (-3, -1.5),
            (-3, -0.5),
            (-3, 0.5),
            (-3, 1.5),  # 4th row (partial)
        ]

        for i, (dx, dy) in enumerate(rack_positions):
            x = rack_center.x + dx * ball_diameter * 0.866
            y = rack_center.y + dy * ball_diameter * 0.5
            balls.append(
                BallState(id=f"ball_{i+1}", position=Vector2D(x, y), number=i + 1)
            )

        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=table,
            game_type=GameType.NINE_BALL,
            is_break=True,
        )

        scenarios.append(
            TestScenario(
                name="break_shot",
                description="Break shot scenario",
                game_state=game_state,
                expected_shot_type=ShotType.BREAK,
                expected_difficulty_range=(0.2, 0.6),
                expected_success_range=(0.6, 0.9),
                expected_problems=[],
                test_assertions=[
                    "Should identify as break shot",
                    "Should recommend high force",
                    "Should target front ball of rack",
                    "Difficulty should be moderate",
                ],
            )
        )

        return scenarios

    def _create_problem_scenarios(self) -> list[TestScenario]:
        """Create scenarios with specific problems."""
        scenarios = []
        table = TableState.standard_9ft_table()

        # Scratch risk scenario
        balls = [
            BallState(id="cue", position=Vector2D(0.1, 0.1), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(0.2, 0.15), number=1),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=table,
            game_type=GameType.PRACTICE,
        )

        scenarios.append(
            TestScenario(
                name="high_scratch_risk",
                description="Shot with high scratch risk due to proximity to pocket",
                game_state=game_state,
                expected_shot_type=ShotType.DIRECT,
                expected_difficulty_range=(0.3, 0.7),
                expected_success_range=(0.2, 0.6),
                expected_problems=["High scratch risk"],
                test_assertions=[
                    "Should identify high scratch risk",
                    "Success probability should be reduced",
                    "Should suggest safety alternatives",
                    "Risk factors should include scratch probability",
                ],
            )
        )

        # Blocked path scenario
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.3), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.5, 0.7), number=1),
            BallState(
                id="ball_2", position=Vector2D(1.0, 0.5), number=2
            ),  # Blocking ball
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=table,
            game_type=GameType.PRACTICE,
        )

        scenarios.append(
            TestScenario(
                name="blocked_direct_path",
                description="Direct path blocked by interfering ball",
                game_state=game_state,
                expected_shot_type=ShotType.COMBINATION,
                expected_difficulty_range=(0.6, 0.9),
                expected_success_range=(0.2, 0.5),
                expected_problems=["Direct path blocked"],
                test_assertions=[
                    "Should identify blocked path",
                    "Should suggest combination or bank shot",
                    "Difficulty should be high",
                    "Should provide alternative approaches",
                ],
            )
        )

        return scenarios

    def _create_advanced_scenarios(self) -> list[TestScenario]:
        """Create advanced strategic scenarios."""
        scenarios = []
        table = TableState.standard_9ft_table()

        # Position play scenario
        balls = [
            BallState(id="cue", position=Vector2D(0.8, 0.4), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.2, 0.5), number=1),
            BallState(id="ball_2", position=Vector2D(1.8, 0.8), number=2),
            BallState(id="ball_3", position=Vector2D(2.0, 0.3), number=3),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=table,
            game_type=GameType.NINE_BALL,
        )

        scenarios.append(
            TestScenario(
                name="position_play_strategic",
                description="Scenario requiring strategic position play",
                game_state=game_state,
                expected_shot_type=ShotType.DIRECT,
                expected_difficulty_range=(0.2, 0.5),
                expected_success_range=(0.5, 0.8),
                expected_problems=[],
                test_assertions=[
                    "Should consider position for next shot",
                    "Should suggest cue ball control techniques",
                    "Strategic advice should mention pattern play",
                    "Safe zones should include position play areas",
                ],
            )
        )

        # Safety play scenario
        balls = [
            BallState(id="cue", position=Vector2D(1.0, 0.8), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(2.0, 0.2), number=1),
            # Cluster of opponent's balls
            BallState(id="ball_2", position=Vector2D(0.5, 0.4), number=2),
            BallState(id="ball_3", position=Vector2D(0.6, 0.5), number=3),
            BallState(id="ball_4", position=Vector2D(0.4, 0.6), number=4),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=table,
            game_type=GameType.EIGHT_BALL,
        )

        scenarios.append(
            TestScenario(
                name="defensive_safety_play",
                description="Scenario where safety play is optimal",
                game_state=game_state,
                expected_shot_type=ShotType.SAFETY,
                expected_difficulty_range=(0.3, 0.6),
                expected_success_range=(0.6, 0.9),
                expected_problems=[],
                test_assertions=[
                    "Should suggest safety play as alternative",
                    "Should identify defensive safe zones",
                    "Should consider opponent's difficult position",
                    "Strategic value should factor in defense",
                ],
            )
        )

        return scenarios

    def run_scenario_test(self, scenario: TestScenario) -> dict[str, Any]:
        """Run a single test scenario and return results."""
        results = {
            "scenario_name": scenario.name,
            "passed": True,
            "errors": [],
            "analysis_results": {},
            "assistance_results": {},
        }

        try:
            # Test shot analysis
            shot_analysis = self.shot_analyzer.analyze_shot(scenario.game_state)
            results["analysis_results"] = {
                "shot_type": shot_analysis.shot_type,
                "difficulty": shot_analysis.difficulty,
                "success_probability": shot_analysis.success_probability,
                "problems": shot_analysis.potential_problems,
                "is_legal": shot_analysis.is_legal,
            }

            # Test predictions
            prediction = self.outcome_predictor.predict_shot_outcome(
                scenario.game_state, shot_analysis
            )
            results["prediction_results"] = {
                "primary_outcome": prediction.primary_outcome.outcome_type,
                "confidence": prediction.overall_confidence,
                "alternatives_count": len(prediction.alternative_outcomes),
            }

            # Test assistance
            assistance = self.assistance_engine.provide_assistance(
                scenario.game_state, AssistanceLevel.INTERMEDIATE
            )
            results["assistance_results"] = {
                "primary_target": assistance.primary_recommendation.ball_id,
                "aiming_confidence": assistance.aiming_guide.confidence,
                "power_level": assistance.power_recommendation.power_level,
                "safe_zones_count": len(assistance.safe_zones),
                "overall_confidence": assistance.confidence,
            }

            # Validate expectations
            self._validate_scenario_expectations(scenario, results)

        except Exception as e:
            results["passed"] = False
            results["errors"].append(f"Exception during testing: {str(e)}")

        return results

    def _validate_scenario_expectations(
        self, scenario: TestScenario, results: dict[str, Any]
    ):
        """Validate that results meet scenario expectations."""
        analysis = results["analysis_results"]

        # Check shot type
        if analysis["shot_type"] != scenario.expected_shot_type:
            results["errors"].append(
                f"Expected shot type {scenario.expected_shot_type}, got {analysis['shot_type']}"
            )
            results["passed"] = False

        # Check difficulty range
        diff_min, diff_max = scenario.expected_difficulty_range
        if not (diff_min <= analysis["difficulty"] <= diff_max):
            results["errors"].append(
                f"Difficulty {analysis['difficulty']:.2f} outside expected range {scenario.expected_difficulty_range}"
            )
            results["passed"] = False

        # Check success probability range
        succ_min, succ_max = scenario.expected_success_range
        if not (succ_min <= analysis["success_probability"] <= succ_max):
            results["errors"].append(
                f"Success probability {analysis['success_probability']:.2f} outside expected range {scenario.expected_success_range}"
            )
            results["passed"] = False

        # Check expected problems
        for expected_problem in scenario.expected_problems:
            if expected_problem not in analysis["problems"]:
                results["errors"].append(
                    f"Expected problem '{expected_problem}' not identified"
                )
                results["passed"] = False

    def run_all_scenarios(self) -> dict[str, Any]:
        """Run all test scenarios and return comprehensive results."""
        scenarios = self.get_all_scenarios()
        results = {
            "total_scenarios": len(scenarios),
            "passed": 0,
            "failed": 0,
            "scenario_results": [],
            "summary": {},
        }

        for scenario in scenarios:
            scenario_result = self.run_scenario_test(scenario)
            results["scenario_results"].append(scenario_result)

            if scenario_result["passed"]:
                results["passed"] += 1
            else:
                results["failed"] += 1

        # Generate summary
        results["summary"] = {
            "pass_rate": results["passed"] / results["total_scenarios"] * 100,
            "critical_failures": [
                result
                for result in results["scenario_results"]
                if not result["passed"] and "critical" in result["scenario_name"]
            ],
            "common_issues": self._analyze_common_issues(results["scenario_results"]),
        }

        return results

    def _analyze_common_issues(
        self, scenario_results: list[dict[str, Any]]
    ) -> dict[str, int]:
        """Analyze common issues across failed scenarios."""
        issue_counts = {}

        for result in scenario_results:
            if not result["passed"]:
                for error in result["errors"]:
                    # Categorize errors
                    if "shot type" in error.lower():
                        issue_counts["shot_type_errors"] = (
                            issue_counts.get("shot_type_errors", 0) + 1
                        )
                    elif "difficulty" in error.lower():
                        issue_counts["difficulty_errors"] = (
                            issue_counts.get("difficulty_errors", 0) + 1
                        )
                    elif "success probability" in error.lower():
                        issue_counts["probability_errors"] = (
                            issue_counts.get("probability_errors", 0) + 1
                        )
                    elif "problem" in error.lower():
                        issue_counts["problem_detection_errors"] = (
                            issue_counts.get("problem_detection_errors", 0) + 1
                        )

        return issue_counts

    def create_custom_scenario(
        self,
        name: str,
        description: str,
        cue_position: Vector2D,
        ball_positions: list[tuple],
        game_type: GameType = GameType.PRACTICE,
    ) -> TestScenario:
        """Create a custom test scenario."""
        table = TableState.standard_9ft_table()
        balls = [BallState(id="cue", position=cue_position, is_cue_ball=True)]

        for i, (x, y, number) in enumerate(ball_positions):
            balls.append(
                BallState(
                    id=f"ball_{number if number else i+1}",
                    position=Vector2D(x, y),
                    number=number,
                )
            )

        game_state = GameState(
            timestamp=0.0, frame_number=0, balls=balls, table=table, game_type=game_type
        )

        return TestScenario(
            name=name,
            description=description,
            game_state=game_state,
            expected_shot_type=ShotType.DIRECT,  # Default
            expected_difficulty_range=(0.0, 1.0),  # Wide range
            expected_success_range=(0.0, 1.0),  # Wide range
            expected_problems=[],
            test_assertions=["Custom scenario - manual validation required"],
        )


def run_comprehensive_tests() -> dict[str, Any]:
    """Run comprehensive analysis tests and return results."""
    test_runner = AnalysisTestScenarios()
    return test_runner.run_all_scenarios()


def validate_analysis_accuracy() -> bool:
    """Quick validation of analysis accuracy."""
    results = run_comprehensive_tests()
    return results["summary"]["pass_rate"] >= 80.0  # 80% pass rate threshold
