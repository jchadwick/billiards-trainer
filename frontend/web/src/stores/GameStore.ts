import { makeAutoObservable, runInAction } from 'mobx';
import type {
  Ball,
  Cue,
  Table,
  GameState,
  Player,
  Shot,
  ShotResult,
  Point2D,
  Rectangle,
  ActionResult,
  GameUpdateMessage
} from './types';

export class GameStore {
  // Observable state
  gameState: GameState = {
    gameType: 'practice',
    currentPlayer: 0,
    players: [],
    isActive: false,
    isPaused: false,
    startTime: null,
    lastShotTime: null,
    shotCount: 0
  };

  balls: Ball[] = [];
  cue: Cue | null = null;
  table: Table = {
    width: 2540, // Standard 9-foot table in mm
    height: 1270,
    pockets: [
      { x: 0, y: 0 }, // Corner pockets
      { x: 1270, y: 0 }, // Side pockets
      { x: 2540, y: 0 },
      { x: 0, y: 1270 },
      { x: 1270, y: 1270 },
      { x: 2540, y: 1270 }
    ],
    rails: [
      { x: 0, y: 0, width: 2540, height: 50 }, // Top rail
      { x: 0, y: 1220, width: 2540, height: 50 }, // Bottom rail
      { x: 0, y: 0, width: 50, height: 1270 }, // Left rail
      { x: 2490, y: 0, width: 50, height: 1270 } // Right rail
    ],
    playArea: { x: 50, y: 50, width: 2440, height: 1170 }
  };

  shotHistory: Shot[] = [];
  private ballsSnapshot: Ball[] = [];

  constructor() {
    makeAutoObservable(this, {}, { autoBind: true });

    this.initializeStandardBalls();
  }

  // Computed values
  get activeBalls(): Ball[] {
    return this.balls.filter(ball => ball.isVisible && !ball.isPocketed);
  }

  get pocketedBalls(): Ball[] {
    return this.balls.filter(ball => ball.isPocketed);
  }

  get cueBall(): Ball | null {
    return this.balls.find(ball => ball.type === 'cue') || null;
  }

  get eightBall(): Ball | null {
    return this.balls.find(ball => ball.type === 'eight') || null;
  }

  get solidBalls(): Ball[] {
    return this.balls.filter(ball => ball.type === 'solid');
  }

  get stripeBalls(): Ball[] {
    return this.balls.filter(ball => ball.type === 'stripe');
  }

  get currentPlayer(): Player | null {
    return this.gameState.players[this.gameState.currentPlayer] || null;
  }

  get isGameOver(): boolean {
    if (!this.gameState.isActive) return false;

    switch (this.gameState.gameType) {
      case 'eightball':
        return this.isEightBallGameOver();
      case 'nineball':
        return this.isNineBallGameOver();
      default:
        return false;
    }
  }

  get winner(): Player | null {
    if (!this.isGameOver) return null;
    // Game-specific winner logic would go here
    return null;
  }

  get ballsInMotion(): Ball[] {
    return this.balls.filter(ball =>
      ball.velocity.x !== 0 || ball.velocity.y !== 0
    );
  }

  get tableIsStill(): boolean {
    return this.ballsInMotion.length === 0;
  }

  get shotInProgress(): boolean {
    return this.ballsInMotion.length > 0;
  }

  // Actions
  async startNewGame(gameType: GameState['gameType'], players: Omit<Player, 'id'>[]): Promise<ActionResult> {
    try {
      const gamePlayers: Player[] = players.map((player, index) => ({
        ...player,
        id: `player_${index + 1}`,
        ballGroup: null,
        score: 0,
        isActive: index === 0
      }));

      runInAction(() => {
        this.gameState = {
          gameType,
          currentPlayer: 0,
          players: gamePlayers,
          isActive: true,
          isPaused: false,
          startTime: new Date(),
          lastShotTime: null,
          shotCount: 0
        };

        this.shotHistory = [];
        this.initializeGameBalls(gameType);
      });

      return {
        success: true,
        data: { gameId: `game_${Date.now()}` },
        timestamp: new Date()
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date()
      };
    }
  }

  pauseGame(): void {
    runInAction(() => {
      this.gameState.isPaused = true;
    });
  }

  resumeGame(): void {
    runInAction(() => {
      this.gameState.isPaused = false;
    });
  }

  endGame(): void {
    runInAction(() => {
      this.gameState.isActive = false;
      this.gameState.isPaused = false;
    });
  }

  updateBalls(newBalls: Ball[]): void {
    runInAction(() => {
      this.balls = [...newBalls];
    });
  }

  updateCue(newCue: Cue | null): void {
    runInAction(() => {
      this.cue = newCue;
    });
  }

  takeBallsSnapshot(): void {
    this.ballsSnapshot = [...this.balls];
  }

  async recordShot(
    playerId: string,
    targetBall: number | null,
    contactPoint: Point2D | null
  ): Promise<ActionResult<Shot>> {
    try {
      const ballsBeforeShot = [...this.ballsSnapshot];
      const ballsAfterShot = [...this.balls];

      const result = this.analyzeShotResult(ballsBeforeShot, ballsAfterShot);

      const shot: Shot = {
        id: `shot_${Date.now()}`,
        timestamp: new Date(),
        playerId,
        cueBallPosition: this.cueBall?.position || { x: 0, y: 0 },
        targetBall,
        contactPoint,
        result,
        ballsBeforeShot,
        ballsAfterShot
      };

      runInAction(() => {
        this.shotHistory.push(shot);
        this.gameState.shotCount++;
        this.gameState.lastShotTime = new Date();

        // Update player scores
        const player = this.gameState.players.find(p => p.id === playerId);
        if (player) {
          player.score += result.points;
        }

        // Handle turn switching
        if (!result.isSuccessful || result.foulCommitted) {
          this.switchToNextPlayer();
        }
      });

      return {
        success: true,
        data: shot,
        timestamp: new Date()
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date()
      };
    }
  }

  handleGameUpdate(message: GameUpdateMessage): void {
    runInAction(() => {
      if (message.data.balls) {
        this.updateBalls(message.data.balls);
      }

      if (message.data.cue !== undefined) {
        this.updateCue(message.data.cue);
      }

      if (message.data.gameState) {
        Object.assign(this.gameState, message.data.gameState);
      }
    });
  }

  resetTable(): void {
    runInAction(() => {
      this.initializeStandardBalls();
      this.cue = null;
      this.shotHistory = [];
    });
  }

  // Ball manipulation actions
  setBallPosition(ballId: number, position: Point2D): void {
    runInAction(() => {
      const ball = this.balls.find(b => b.id === ballId);
      if (ball) {
        ball.position = { ...position };
        ball.velocity = { x: 0, y: 0 };
      }
    });
  }

  setBallPocketed(ballId: number, isPocketed: boolean): void {
    runInAction(() => {
      const ball = this.balls.find(b => b.id === ballId);
      if (ball) {
        ball.isPocketed = isPocketed;
        ball.isVisible = !isPocketed;
        if (isPocketed) {
          ball.velocity = { x: 0, y: 0 };
        }
      }
    });
  }

  setBallVisible(ballId: number, isVisible: boolean): void {
    runInAction(() => {
      const ball = this.balls.find(b => b.id === ballId);
      if (ball) {
        ball.isVisible = isVisible;
      }
    });
  }

  // Private methods
  private initializeStandardBalls(): void {
    const standardBalls: Ball[] = [
      // Cue ball
      { id: 0, type: 'cue', color: 'white', position: { x: 635, y: 635 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },

      // Solid balls (1-7)
      { id: 1, type: 'solid', color: 'yellow', position: { x: 1905, y: 635 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 2, type: 'solid', color: 'blue', position: { x: 1940, y: 615 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 3, type: 'solid', color: 'red', position: { x: 1940, y: 655 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 4, type: 'solid', color: 'purple', position: { x: 1975, y: 595 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 5, type: 'solid', color: 'orange', position: { x: 1975, y: 635 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 6, type: 'solid', color: 'green', position: { x: 1975, y: 675 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 7, type: 'solid', color: 'maroon', position: { x: 2010, y: 575 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },

      // Eight ball
      { id: 8, type: 'eight', color: 'black', position: { x: 2010, y: 635 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },

      // Stripe balls (9-15)
      { id: 9, type: 'stripe', color: 'yellow-stripe', position: { x: 2010, y: 695 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 10, type: 'stripe', color: 'blue-stripe', position: { x: 2045, y: 555 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 11, type: 'stripe', color: 'red-stripe', position: { x: 2045, y: 595 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 12, type: 'stripe', color: 'purple-stripe', position: { x: 2045, y: 635 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 13, type: 'stripe', color: 'orange-stripe', position: { x: 2045, y: 675 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 14, type: 'stripe', color: 'green-stripe', position: { x: 2045, y: 715 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 15, type: 'stripe', color: 'maroon-stripe', position: { x: 2080, y: 535 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 }
    ];

    this.balls = standardBalls;
  }

  private initializeGameBalls(gameType: GameState['gameType']): void {
    switch (gameType) {
      case 'eightball':
        this.initializeStandardBalls();
        break;
      case 'nineball':
        this.initializeNineBalls();
        break;
      case 'straight':
        this.initializeStandardBalls();
        break;
      case 'practice':
        this.initializeStandardBalls();
        break;
    }
  }

  private initializeNineBalls(): void {
    const nineBalls: Ball[] = [
      // Cue ball
      { id: 0, type: 'cue', color: 'white', position: { x: 635, y: 635 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },

      // Balls 1-9
      { id: 1, type: 'solid', color: 'yellow', position: { x: 1905, y: 635 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 2, type: 'solid', color: 'blue', position: { x: 1940, y: 615 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 3, type: 'solid', color: 'red', position: { x: 1940, y: 655 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 4, type: 'solid', color: 'purple', position: { x: 1975, y: 595 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 5, type: 'solid', color: 'orange', position: { x: 1975, y: 635 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 6, type: 'solid', color: 'green', position: { x: 1975, y: 675 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 7, type: 'solid', color: 'maroon', position: { x: 2010, y: 615 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 8, type: 'solid', color: 'black', position: { x: 2010, y: 655 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 },
      { id: 9, type: 'solid', color: 'yellow-stripe', position: { x: 2045, y: 635 }, velocity: { x: 0, y: 0 }, isVisible: true, isPocketed: false, confidence: 1.0 }
    ];

    this.balls = nineBalls;
  }

  private analyzeShotResult(ballsBefore: Ball[], ballsAfter: Ball[]): ShotResult {
    const ballsPocketed: number[] = [];

    // Find balls that were pocketed
    for (const ballBefore of ballsBefore) {
      const ballAfter = ballsAfter.find(b => b.id === ballBefore.id);
      if (ballAfter && !ballBefore.isPocketed && ballAfter.isPocketed) {
        ballsPocketed.push(ballBefore.id);
      }
    }

    // Basic foul detection (simplified)
    const foulCommitted = this.detectFouls(ballsBefore, ballsAfter, ballsPocketed);

    // Calculate points based on game type
    const points = this.calculatePoints(ballsPocketed, foulCommitted);

    const isSuccessful = ballsPocketed.length > 0 && !foulCommitted;

    return {
      ballsPocketed,
      foulCommitted,
      foulType: foulCommitted ? 'generic' : null,
      isSuccessful,
      points
    };
  }

  private detectFouls(ballsBefore: Ball[], ballsAfter: Ball[], ballsPocketed: number[]): boolean {
    // Simplified foul detection
    // In a real implementation, this would be much more sophisticated

    // Cue ball pocketed
    if (ballsPocketed.includes(0)) {
      return true;
    }

    // No balls hit (would require more complex physics analysis)
    // For now, assume if no balls moved significantly, it's a foul

    return false;
  }

  private calculatePoints(ballsPocketed: number[], foulCommitted: boolean): number {
    if (foulCommitted) return 0;

    // Basic point calculation
    return ballsPocketed.length;
  }

  private switchToNextPlayer(): void {
    const nextPlayerIndex = (this.gameState.currentPlayer + 1) % this.gameState.players.length;

    runInAction(() => {
      // Deactivate current player
      this.gameState.players[this.gameState.currentPlayer].isActive = false;

      // Activate next player
      this.gameState.currentPlayer = nextPlayerIndex;
      this.gameState.players[nextPlayerIndex].isActive = true;
    });
  }

  private isEightBallGameOver(): boolean {
    const eightBall = this.eightBall;
    return eightBall ? eightBall.isPocketed : false;
  }

  private isNineBallGameOver(): boolean {
    const nineBall = this.balls.find(ball => ball.id === 9);
    return nineBall ? nineBall.isPocketed : false;
  }
}
