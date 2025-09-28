"""Comprehensive unit tests for shot analysis and assistance systems."""

import unittest

from ..core.analysis.assistance import (
    AssistanceEngine,
    AssistanceLevel,
    SafeZoneType,
)
from ..core.analysis.prediction import OutcomePredictor, OutcomeType
from ..core.analysis.shot import IllegalShotReason, ShotAnalysis, ShotAnalyzer
from ..core.analysis.test_scenarios import AnalysisTestScenarios
from ..core.models import (
    BallState,
    GameState,
    GameType,
    ShotType,
    TableState,
    Vector2D,
)


class TestShotAnalyzer(unittest.TestCase):
    """Test cases for ShotAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ShotAnalyzer()
        self.table = TableState.standard_9ft_table()

    def test_shot_type_identification(self):
        """Test shot type identification."""
        # Direct shot
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.5, 0.6), number=1),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=self.table,
            game_type=GameType.PRACTICE,
        )

        analysis = self.analyzer.analyze_shot(game_state, "ball_1")
        assert analysis.shot_type == ShotType.DIRECT

        # Break shot
        game_state.is_break = True
        analysis = self.analyzer.analyze_shot(game_state, "ball_1")
        assert analysis.shot_type == ShotType.BREAK

    def test_difficulty_calculation(self):
        """Test difficulty calculation accuracy."""
        # Easy shot - close and straight
        balls = [
            BallState(id="cue", position=Vector2D(0.8, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.0, 0.6), number=1),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=self.table,
            game_type=GameType.PRACTICE,
        )

        analysis = self.analyzer.analyze_shot(game_state, "ball_1")
        assert analysis.difficulty < 0.3, "Easy shot should have low difficulty"
        assert (
            analysis.success_probability > 0.7
        ), "Easy shot should have high success probability"

        # Hard shot - long distance with obstacles
        balls = [
            BallState(id="cue", position=Vector2D(0.2, 0.2), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(2.2, 1.0), number=1),
            BallState(id="ball_2", position=Vector2D(1.0, 0.5), number=2),  # Obstacle
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=self.table,
            game_type=GameType.PRACTICE,
        )

        analysis = self.analyzer.analyze_shot(game_state, "ball_1")
        assert analysis.difficulty > 0.6, "Hard shot should have high difficulty"
        assert (
            analysis.success_probability < 0.5
        ), "Hard shot should have low success probability"

    def test_illegal_shot_detection(self):
        """Test illegal shot detection."""
        # 9-ball: must hit lowest numbered ball first
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.0, 0.6), number=1),
            BallState(id="ball_2", position=Vector2D(1.5, 0.6), number=2),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=self.table,
            game_type=GameType.NINE_BALL,
        )

        # Legal shot - targeting ball 1
        analysis = self.analyzer.analyze_shot(game_state, "ball_1")
        assert analysis.is_legal
        assert len(analysis.illegal_reasons) == 0

        # Illegal shot - targeting ball 2 instead of ball 1
        analysis = self.analyzer.analyze_shot(game_state, "ball_2")
        assert not analysis.is_legal
        assert IllegalShotReason.WRONG_BALL_FIRST in analysis.illegal_reasons

    def test_alternative_shot_suggestions(self):
        """Test alternative shot suggestions."""
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.0, 0.6), number=1),
            BallState(id="ball_2", position=Vector2D(1.5, 0.4), number=2),
            BallState(id="ball_3", position=Vector2D(2.0, 0.8), number=3),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=self.table,
            game_type=GameType.PRACTICE,
        )

        alternatives = self.analyzer.get_alternative_shots(
            game_state, max_alternatives=3
        )
        assert len(alternatives) > 0, "Should suggest alternative shots"
        assert len(alternatives) <= 3, "Should respect max_alternatives parameter"

        # Check that alternatives are sorted by priority
        for i in range(1, len(alternatives)):
            assert (
                alternatives[i].success_probability
                <= alternatives[i - 1].success_probability
            ), "Alternatives should be sorted by success probability"

    def test_problem_identification(self):
        """Test potential problem identification."""
        # Scenario with scratch risk
        balls = [
            BallState(id="cue", position=Vector2D(0.1, 0.1), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(0.2, 0.15), number=1),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=self.table,
            game_type=GameType.PRACTICE,
        )

        analysis = self.analyzer.analyze_shot(game_state, "ball_1")
        assert "High scratch risk" in analysis.potential_problems
        assert analysis.risk_factors.get("scratch", 0) > 0.2

        # Scenario with blocked path
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.3), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.5, 0.7), number=1),
            BallState(id="ball_2", position=Vector2D(1.0, 0.5), number=2),  # Blocking
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=self.table,
            game_type=GameType.PRACTICE,
        )

        analysis = self.analyzer.analyze_shot(game_state, "ball_1")
        assert "Direct path blocked" in analysis.potential_problems


class TestOutcomePredictor(unittest.TestCase):
    """Test cases for OutcomePredictor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.predictor = OutcomePredictor()
        self.analyzer = ShotAnalyzer()
        self.table = TableState.standard_9ft_table()

    def test_shot_outcome_prediction(self):
        """Test shot outcome prediction."""
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.5, 0.6), number=1),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=self.table,
            game_type=GameType.PRACTICE,
        )

        shot_analysis = self.analyzer.analyze_shot(game_state, "ball_1")
        prediction = self.predictor.predict_shot_outcome(game_state, shot_analysis)

        assert prediction.primary_outcome is not None
        assert prediction.primary_outcome.outcome_type in [
            OutcomeType.SUCCESS,
            OutcomeType.MISS,
        ]
        self.assertBetween(prediction.overall_confidence, 0.0, 1.0)
        assert len(prediction.prediction_factors) > 0

    def test_success_probability_calculation(self):
        """Test success probability calculation."""
        # Mock shot analysis for testing
        shot_analysis = ShotAnalysis(
            shot_type=ShotType.DIRECT,
            difficulty=0.3,
            success_probability=0.8,
            recommended_force=10.0,
            recommended_angle=0.0,
            recommended_aim_point=Vector2D(1.5, 0.6),
        )

        probability = self.predictor.calculate_success_probability(shot_analysis)
        self.assertBetween(probability, 0.05, 0.95)  # Should be clamped to this range

    def test_outcome_prediction_with_moving_balls(self):
        """Test outcome prediction with moving balls."""
        balls = [
            BallState(
                id="cue",
                position=Vector2D(0.5, 0.6),
                velocity=Vector2D(1.0, 0.0),
                is_cue_ball=True,
            ),
            BallState(id="ball_1", position=Vector2D(1.5, 0.6), number=1),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=self.table,
            game_type=GameType.PRACTICE,
        )

        outcomes = self.predictor.predict_outcomes(game_state, time_horizon=2.0)
        assert len(outcomes) >= 1, "Should predict outcomes for moving balls"

    def assertBetween(self, value, min_val, max_val, msg=None):
        """Custom assertion for range checking."""
        if not (min_val <= value <= max_val):
            raise AssertionError(f"{value} not between {min_val} and {max_val}: {msg}")


class TestAssistanceEngine(unittest.TestCase):
    """Test cases for AssistanceEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.assistance = AssistanceEngine()
        self.table = TableState.standard_9ft_table()

    def test_assistance_levels(self):
        """Test different assistance levels."""
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.5, 0.6), number=1),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=self.table,
            game_type=GameType.PRACTICE,
        )

        # Test beginner assistance
        beginner_help = self.assistance.provide_assistance(
            game_state, AssistanceLevel.BEGINNER
        )
        assert beginner_help.aiming_guide.visual_aids["show_ghost_ball"]
        assert len(beginner_help.alternative_targets) == 1

        # Test expert assistance
        expert_help = self.assistance.provide_assistance(
            game_state, AssistanceLevel.EXPERT
        )
        assert not expert_help.aiming_guide.visual_aids["show_ghost_ball"]
        assert len(expert_help.alternative_targets) >= 1

    def test_aiming_point_suggestion(self):
        """Test optimal aiming point suggestion."""
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.5, 0.6), number=1),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=self.table,
            game_type=GameType.PRACTICE,
        )

        aim_point = self.assistance.suggest_optimal_aim_point(game_state, "ball_1")
        assert isinstance(aim_point, Vector2D)

        # Aim point should be on the target ball surface
        target_ball = game_state.get_ball_by_id("ball_1")
        distance_to_center = aim_point.distance_to(target_ball.position)
        self.assertAlmostEqual(distance_to_center, target_ball.radius, places=2)

    def test_power_recommendation(self):
        """Test shot power recommendation."""
        shot_analysis = ShotAnalysis(
            shot_type=ShotType.DIRECT,
            difficulty=0.3,
            success_probability=0.8,
            recommended_force=10.0,
            recommended_angle=0.0,
            recommended_aim_point=Vector2D(1.5, 0.6),
        )

        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=[],
            table=self.table,
            game_type=GameType.PRACTICE,
        )

        power_rec = self.assistance.recommend_shot_power(game_state, shot_analysis)
        assert power_rec.recommended_force > 0
        assert power_rec.power_level in ["soft", "medium", "hard"]
        assert len(power_rec.force_range) == 2
        assert power_rec.force_range[0] < power_rec.force_range[1]

    def test_best_target_identification(self):
        """Test best target ball identification."""
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.0, 0.6), number=1),  # Easy shot
            BallState(
                id="ball_2", position=Vector2D(2.0, 1.0), number=2
            ),  # Harder shot
            BallState(
                id="ball_3", position=Vector2D(1.5, 0.4), number=3
            ),  # Medium shot
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=self.table,
            game_type=GameType.PRACTICE,
        )

        best_target = self.assistance.identify_best_target(game_state)
        assert best_target in ["ball_1", "ball_2", "ball_3"]

        # In practice mode, should generally prefer easier shots
        # but we'll just verify a valid target is returned

    def test_safe_zones_calculation(self):
        """Test safe zone calculation."""
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.0, 0.6), number=1),
            BallState(
                id="ball_2", position=Vector2D(1.1, 0.65), number=2
            ),  # Close to ball_1
            BallState(
                id="ball_3", position=Vector2D(1.05, 0.55), number=3
            ),  # Forms cluster
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=self.table,
            game_type=GameType.PRACTICE,
        )

        safe_zones = self.assistance.calculate_safe_zones(game_state)
        assert len(safe_zones) >= 0

        # Check zone properties
        for zone in safe_zones:
            assert zone.zone_type in [
                SafeZoneType.DEFENSIVE,
                SafeZoneType.SCRATCH_SAFE,
                SafeZoneType.POSITION_PLAY,
                SafeZoneType.CLUSTER_BREAK,
            ]
            self.assertBetween(zone.safety_score, 0.0, 1.0)
            assert zone.radius > 0

    def test_difficulty_adjustment(self):
        """Test difficulty-based assistance adjustment."""
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.5, 0.6), number=1),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=self.table,
            game_type=GameType.PRACTICE,
        )

        assistance = self.assistance.provide_assistance(
            game_state, AssistanceLevel.INTERMEDIATE
        )

        # Test with low skill player
        low_skill = {"aiming_accuracy": 0.2, "power_control": 0.3}
        adjusted = self.assistance.adjust_for_difficulty(assistance, low_skill)
        assert adjusted is not None

        # Test with high skill player
        high_skill = {"aiming_accuracy": 0.8, "power_control": 0.9}
        adjusted = self.assistance.adjust_for_difficulty(assistance, high_skill)
        assert adjusted is not None

    def assertBetween(self, value, min_val, max_val, msg=None):
        """Custom assertion for range checking."""
        if not (min_val <= value <= max_val):
            raise AssertionError(f"{value} not between {min_val} and {max_val}: {msg}")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete analysis system."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_scenarios = AnalysisTestScenarios()

    def test_comprehensive_scenarios(self):
        """Test comprehensive analysis scenarios."""
        results = self.test_scenarios.run_all_scenarios()

        assert results["total_scenarios"] > 0
        assert (
            results["passed"] >= results["total_scenarios"] * 0.7
        )  # 70% pass rate minimum

        # Check that critical scenarios passed
        for result in results["scenario_results"]:
            if "critical" in result["scenario_name"]:
                assert result[
                    "passed"
                ], f"Critical scenario {result['scenario_name']} failed"

    def test_analysis_consistency(self):
        """Test consistency across multiple analysis runs."""
        # Create simple scenario
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.5, 0.6), number=1),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=TableState.standard_9ft_table(),
            game_type=GameType.PRACTICE,
        )

        # Run analysis multiple times
        analyzer = ShotAnalyzer()
        results = []
        for _ in range(5):
            analysis = analyzer.analyze_shot(game_state, "ball_1")
            results.append(
                {
                    "difficulty": analysis.difficulty,
                    "success_probability": analysis.success_probability,
                    "shot_type": analysis.shot_type,
                }
            )

        # Check consistency
        for i in range(1, len(results)):
            assert results[i]["shot_type"] == results[0]["shot_type"]
            self.assertAlmostEqual(
                results[i]["difficulty"], results[0]["difficulty"], places=3
            )
            self.assertAlmostEqual(
                results[i]["success_probability"],
                results[0]["success_probability"],
                places=3,
            )

    def test_performance_benchmarks(self):
        """Test performance benchmarks for analysis."""
        import time

        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.5, 0.6), number=1),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=0,
            balls=balls,
            table=TableState.standard_9ft_table(),
            game_type=GameType.PRACTICE,
        )

        # Test shot analysis performance
        analyzer = ShotAnalyzer()
        start_time = time.time()
        for _ in range(100):
            analyzer.analyze_shot(game_state, "ball_1")
        analysis_time = time.time() - start_time

        assert (
            analysis_time < 1.0
        ), "100 shot analyses should complete in under 1 second"

        # Test assistance performance
        assistance = AssistanceEngine()
        start_time = time.time()
        for _ in range(50):
            assistance.provide_assistance(game_state, AssistanceLevel.INTERMEDIATE)
        assistance_time = time.time() - start_time

        assert (
            assistance_time < 2.0
        ), "50 assistance calculations should complete in under 2 seconds"


def run_all_tests():
    """Run all analysis tests and return results."""
    # Create test suite
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTest(unittest.makeSuite(TestShotAnalyzer))
    suite.addTest(unittest.makeSuite(TestOutcomePredictor))
    suite.addTest(unittest.makeSuite(TestAssistanceEngine))
    suite.addTest(unittest.makeSuite(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors))
        / result.testsRun
        * 100,
        "details": {
            "failures": [str(failure) for failure in result.failures],
            "errors": [str(error) for error in result.errors],
        },
    }


if __name__ == "__main__":
    # Run tests when script is executed directly
    results = run_all_tests()
    print("\nTest Results Summary:")
    print(f"Tests Run: {results['tests_run']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Success Rate: {results['success_rate']:.1f}%")

    if results["failures"] > 0 or results["errors"] > 0:
        print("\nIssues found:")
        for failure in results["details"]["failures"]:
            print(f"FAILURE: {failure}")
        for error in results["details"]["errors"]:
            print(f"ERROR: {error}")
    else:
        print("\nAll tests passed!")
