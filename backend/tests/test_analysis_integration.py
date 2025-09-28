"""Integration test for the complete shot analysis and recommendation system."""

import os
import sys
import unittest

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ..core.analysis.assistance import AssistanceEngine, AssistanceLevel
from ..core.analysis.prediction import OutcomePredictor
from ..core.analysis.shot import ShotAnalyzer
from ..core.models import (
    BallState,
    GameState,
    GameType,
    ShotType,
    TableState,
    Vector2D,
)


class TestAnalysisIntegration(unittest.TestCase):
    """Integration test for the complete analysis system."""

    def setUp(self):
        """Set up the complete analysis pipeline."""
        self.shot_analyzer = ShotAnalyzer()
        self.outcome_predictor = OutcomePredictor()
        self.assistance_engine = AssistanceEngine()
        self.table = TableState.standard_9ft_table()

    def test_complete_analysis_pipeline(self):
        """Test the complete analysis pipeline from game state to recommendations."""
        # Create a realistic game scenario
        balls = [
            BallState(id="cue", position=Vector2D(0.6, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.2, 0.7), number=1),
            BallState(id="ball_2", position=Vector2D(1.8, 0.4), number=2),
            BallState(id="ball_3", position=Vector2D(2.0, 0.9), number=3),
            BallState(id="ball_4", position=Vector2D(1.0, 0.3), number=4),
        ]

        game_state = GameState(
            timestamp=0.0,
            frame_number=1,
            balls=balls,
            table=self.table,
            game_type=GameType.NINE_BALL,
        )

        print("\n=== Testing Complete Analysis Pipeline ===")
        print(f"Game type: {game_state.game_type.value}")
        print(f"Balls on table: {len([b for b in balls if not b.is_pocketed])}")

        # Step 1: Shot Analysis
        print("\n1. Shot Analysis...")
        shot_analysis = self.shot_analyzer.analyze_shot(game_state)

        assert shot_analysis is not None
        assert shot_analysis.target_ball_id is not None
        assert shot_analysis.shot_type in list(ShotType)
        self.assertBetween(shot_analysis.difficulty, 0.0, 1.0)
        self.assertBetween(shot_analysis.success_probability, 0.0, 1.0)

        print(f"  Target: {shot_analysis.target_ball_id}")
        print(f"  Type: {shot_analysis.shot_type.value}")
        print(f"  Difficulty: {shot_analysis.difficulty:.2f}")
        print(f"  Success probability: {shot_analysis.success_probability:.2f}")
        print(f"  Legal shot: {shot_analysis.is_legal}")
        print(f"  Problems: {shot_analysis.potential_problems}")

        # Step 2: Outcome Prediction
        print("\n2. Outcome Prediction...")
        prediction = self.outcome_predictor.predict_shot_outcome(
            game_state, shot_analysis
        )

        assert prediction is not None
        assert prediction.primary_outcome is not None
        self.assertBetween(prediction.overall_confidence, 0.0, 1.0)

        print(f"  Primary outcome: {prediction.primary_outcome.outcome_type.value}")
        print(f"  Confidence: {prediction.overall_confidence:.2f}")
        print(f"  Alternatives: {len(prediction.alternative_outcomes)}")

        # Step 3: Player Assistance
        print("\n3. Player Assistance...")

        # Test different assistance levels
        for level in [
            AssistanceLevel.BEGINNER,
            AssistanceLevel.INTERMEDIATE,
            AssistanceLevel.EXPERT,
        ]:
            print(f"\n  Testing {level.value} assistance...")
            assistance = self.assistance_engine.provide_assistance(game_state, level)

            assert assistance is not None
            assert assistance.assistance_level == level
            assert assistance.primary_recommendation is not None
            assert assistance.aiming_guide is not None
            assert assistance.power_recommendation is not None

            print(f"    Target: {assistance.primary_recommendation.ball_id}")
            print(f"    Cut angle: {assistance.aiming_guide.cut_angle:.1f}°")
            print(f"    Power level: {assistance.power_recommendation.power_level}")
            print(f"    Safe zones: {len(assistance.safe_zones)}")
            print(f"    Confidence: {assistance.confidence:.2f}")

        # Step 4: Alternative Analysis
        print("\n4. Alternative Shots...")
        alternatives = self.shot_analyzer.get_alternative_shots(
            game_state, max_alternatives=3
        )

        assert len(alternatives) <= 3

        for i, alt in enumerate(alternatives):
            print(
                f"  Alt {i+1}: Ball {alt.ball_id}, "
                f"Success: {alt.success_probability:.2f}, "
                f"Difficulty: {alt.shot_analysis.difficulty:.2f}"
            )

        print("\n=== Pipeline Test Complete ===")
        print("✓ Shot analysis working")
        print("✓ Outcome prediction working")
        print("✓ Player assistance working")
        print("✓ Alternative analysis working")

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n=== Testing Edge Cases ===")

        # Test with minimal balls
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.5, 0.6), number=1),
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=1,
            balls=balls,
            table=self.table,
            game_type=GameType.PRACTICE,
        )

        # Should work with minimal scenario
        shot_analysis = self.shot_analyzer.analyze_shot(game_state)
        assert shot_analysis.target_ball_id == "ball_1"

        assistance = self.assistance_engine.provide_assistance(game_state)
        assert assistance.primary_recommendation.ball_id == "ball_1"

        print("✓ Minimal ball scenario handled")

        # Test with ball very close to pocket (scratch risk)
        balls = [
            BallState(id="cue", position=Vector2D(0.05, 0.05), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(0.1, 0.1), number=1),
        ]
        game_state.balls = balls

        shot_analysis = self.shot_analyzer.analyze_shot(game_state)
        assert "High scratch risk" in shot_analysis.potential_problems

        print("✓ Scratch risk detection working")

        # Test with clustered balls
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.5, 0.6), number=1),
            BallState(
                id="ball_2", position=Vector2D(1.55, 0.62), number=2
            ),  # Very close
            BallState(id="ball_3", position=Vector2D(1.52, 0.58), number=3),  # Cluster
        ]
        game_state.balls = balls

        assistance = self.assistance_engine.provide_assistance(game_state)
        # Should handle clustered balls without crashing
        assert assistance is not None

        print("✓ Clustered balls scenario handled")

    def test_game_type_specific_behavior(self):
        """Test game type specific behavior."""
        print("\n=== Testing Game-Specific Behavior ===")

        # Test 9-ball behavior - must hit lowest numbered ball
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.6), is_cue_ball=True),
            BallState(id="ball_1", position=Vector2D(1.0, 0.6), number=1),
            BallState(id="ball_2", position=Vector2D(1.5, 0.6), number=2),
            BallState(id="ball_3", position=Vector2D(2.0, 0.6), number=3),
        ]

        game_state = GameState(
            timestamp=0.0,
            frame_number=1,
            balls=balls,
            table=self.table,
            game_type=GameType.NINE_BALL,
        )

        assistance = self.assistance_engine.provide_assistance(game_state)
        assert assistance.primary_recommendation.ball_id == "ball_1"

        print("✓ 9-ball lowest ball targeting working")

        # Test practice mode - can target any ball
        game_state.game_type = GameType.PRACTICE
        assistance = self.assistance_engine.provide_assistance(game_state)
        # Should work and target some ball
        assert assistance.primary_recommendation.ball_id is not None

        print("✓ Practice mode flexibility working")

    def test_performance_requirements(self):
        """Test that analysis meets performance requirements."""
        import time

        print("\n=== Testing Performance Requirements ===")

        # Create a complex scenario
        balls = [BallState(id="cue", position=Vector2D(0.5, 0.6), is_cue_ball=True)]

        # Add many balls to test performance
        for i in range(1, 15):
            x = 0.8 + (i % 5) * 0.3
            y = 0.3 + (i // 5) * 0.2
            balls.append(BallState(id=f"ball_{i}", position=Vector2D(x, y), number=i))

        game_state = GameState(
            timestamp=0.0,
            frame_number=1,
            balls=balls,
            table=self.table,
            game_type=GameType.PRACTICE,
        )

        # Test shot analysis performance
        start_time = time.time()
        for _ in range(10):
            self.shot_analyzer.analyze_shot(game_state)
        analysis_time = time.time() - start_time

        assert (
            analysis_time < 0.5
        ), "10 shot analyses should complete in under 0.5 seconds"
        print(f"✓ Shot analysis: {analysis_time*100:.1f}ms per analysis")

        # Test assistance performance
        start_time = time.time()
        for _ in range(5):
            self.assistance_engine.provide_assistance(
                game_state, AssistanceLevel.INTERMEDIATE
            )
        assistance_time = time.time() - start_time

        assert (
            assistance_time < 1.0
        ), "5 assistance calculations should complete in under 1 second"
        print(f"✓ Assistance: {assistance_time*200:.1f}ms per assistance")

    def test_accuracy_benchmarks(self):
        """Test accuracy against known scenarios."""
        print("\n=== Testing Accuracy Benchmarks ===")

        # Easy straight shot - should have high success probability
        balls = [
            BallState(id="cue", position=Vector2D(0.5, 0.6), is_cue_ball=True),
            BallState(
                id="ball_1", position=Vector2D(0.8, 0.6), number=1
            ),  # Very close, straight
        ]
        game_state = GameState(
            timestamp=0.0,
            frame_number=1,
            balls=balls,
            table=self.table,
            game_type=GameType.PRACTICE,
        )

        shot_analysis = self.shot_analyzer.analyze_shot(game_state)
        assert (
            shot_analysis.success_probability > 0.7
        ), "Easy shot should have high success probability"
        assert shot_analysis.difficulty < 0.4, "Easy shot should have low difficulty"
        print("✓ Easy shot accuracy validated")

        # Difficult shot - should have lower success probability
        balls = [
            BallState(id="cue", position=Vector2D(0.2, 0.2), is_cue_ball=True),
            BallState(
                id="ball_1", position=Vector2D(2.2, 1.0), number=1
            ),  # Far, near corner
            BallState(id="ball_2", position=Vector2D(1.0, 0.5), number=2),  # Blocking
        ]
        game_state.balls = balls

        shot_analysis = self.shot_analyzer.analyze_shot(game_state)
        assert (
            shot_analysis.success_probability < 0.6
        ), "Difficult shot should have lower success probability"
        assert (
            shot_analysis.difficulty > 0.5
        ), "Difficult shot should have higher difficulty"
        print("✓ Difficult shot accuracy validated")

    def assertBetween(self, value, min_val, max_val, msg=None):
        """Custom assertion for range checking."""
        if not (min_val <= value <= max_val):
            raise AssertionError(f"{value} not between {min_val} and {max_val}: {msg}")


def main():
    """Run the integration tests."""
    print("=" * 60)
    print("BILLIARDS TRAINER - SHOT ANALYSIS INTEGRATION TEST")
    print("=" * 60)

    # Run the integration tests
    unittest.main(verbosity=2, exit=False)

    print("\n" + "=" * 60)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
