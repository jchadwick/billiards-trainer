import React, { useState, useRef, useEffect, useCallback } from "react";
import { observer } from "mobx-react-lite";
import { Card, CardContent, Button } from "../../ui";
import { apiClient } from "../../../api/client";
import type { CalibrationData } from "../../../types/video";

const CALIBRATION_CANVAS_WIDTH = 640;
const CALIBRATION_CANVAS_HEIGHT = 360;

// Types for calibration workflow
interface Point {
  x: number;
  y: number;
  id: string;
  screenX: number;
  screenY: number;
  worldX: number;
  worldY: number;
}

interface MarkerDot {
  id: string;
  x: number;
  y: number;
}

// Interactive video feed with point selection
const VideoFeedCanvas: React.FC<{
  onPointSelect: (x: number, y: number) => void;
  points: Point[];
  markerDots: MarkerDot[];
  width: number;
  height: number;
  overlayVisible: boolean;
  savedCalibrationData?: CalibrationData | null;
  selectionMode?: 'corners' | 'markers';
}> = ({
  onPointSelect,
  points,
  markerDots,
  width,
  height,
  overlayVisible,
  savedCalibrationData,
  selectionMode = 'corners',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationFrameRef = useRef<number>();
  const [streamConnected, setStreamConnected] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);

  // Get the video stream URL from the API client
  const videoStreamUrl = apiClient.getVideoStreamUrl();

  // Continuous canvas drawing loop for live video
  const drawFrame = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw video frame if available
    if (img.complete && img.naturalWidth > 0) {
      ctx.drawImage(img, 0, 0, width, height);
    }

    // Draw saved calibration overlay in magenta/purple (if exists)
    if (savedCalibrationData && savedCalibrationData.isValid && savedCalibrationData.corners.length >= 4) {
      // Draw saved calibration boundary lines
      ctx.strokeStyle = "#d946ef"; // Magenta
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 4]); // Dashed line

      ctx.beginPath();
      savedCalibrationData.corners.forEach((corner, index) => {
        const x = corner.screenPosition.x;
        const y = corner.screenPosition.y;
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]); // Reset dash pattern

      // Draw saved calibration corner points
      savedCalibrationData.corners.forEach((corner, index) => {
        const x = corner.screenPosition.x;
        const y = corner.screenPosition.y;

        // Outer glow
        ctx.fillStyle = "rgba(217, 70, 239, 0.3)"; // Magenta with transparency
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, 2 * Math.PI);
        ctx.fill();

        // Main point
        ctx.fillStyle = "#d946ef"; // Magenta
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();

        // Label
        ctx.fillStyle = "#d946ef";
        ctx.strokeStyle = "#000000";
        ctx.lineWidth = 3;
        ctx.font = "bold 11px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        const label = `S${index + 1}`; // "S" for "Saved"
        ctx.strokeText(label, x, y - 18);
        ctx.fillText(label, x, y - 18);
      });

      // Draw "Saved Calibration" badge
      ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
      ctx.fillRect(10, 10, 150, 30);
      ctx.strokeStyle = "#d946ef";
      ctx.lineWidth = 2;
      ctx.strokeRect(10, 10, 150, 30);

      ctx.fillStyle = "#d946ef";
      ctx.font = "bold 12px sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText("✓ Saved Calibration", 18, 25);
    }

    // Draw calibration points overlay if visible (new points being set)
    if (overlayVisible) {
      points.forEach((point, index) => {
        // Draw crosshair lines (X and Y axis through the point)
        ctx.strokeStyle = "rgba(59, 130, 246, 0.5)"; // Semi-transparent blue
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]); // Dashed line

        // Vertical line (Y-axis)
        ctx.beginPath();
        ctx.moveTo(point.screenX, 0);
        ctx.lineTo(point.screenX, height);
        ctx.stroke();

        // Horizontal line (X-axis)
        ctx.beginPath();
        ctx.moveTo(0, point.screenY);
        ctx.lineTo(width, point.screenY);
        ctx.stroke();

        ctx.setLineDash([]); // Reset dash pattern

        ctx.fillStyle = "#3b82f6"; // Blue
        ctx.strokeStyle = "#1e40af"; // Darker blue
        ctx.lineWidth = 2;

        // Draw point circle
        ctx.beginPath();
        ctx.arc(point.screenX, point.screenY, 8, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();

        // Draw point label
        ctx.fillStyle = "#ffffff";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText((index + 1).toString(), point.screenX, point.screenY + 4);
      });

      // Draw connecting lines for table outline
      if (points.length >= 4) {
        ctx.strokeStyle = "#3b82f6"; // Blue
        ctx.lineWidth = 3;
        ctx.beginPath();
        points.forEach((point, index) => {
          if (index === 0) {
            ctx.moveTo(point.screenX, point.screenY);
          } else {
            ctx.lineTo(point.screenX, point.screenY);
          }
        });
        ctx.closePath();
        ctx.stroke();
      }

      // Add click instruction text at bottom
      ctx.fillStyle = "#9ca3af";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      const instructionText = selectionMode === 'markers'
        ? "Click on table marker dots (spots/markings) to mark them for masking"
        : "Click on the 4 corners of the playing area to set them manually";
      ctx.fillText(instructionText, width / 2, height - 20);
    }

    // Draw marker dots (table spots/markings to be masked)
    if (markerDots.length > 0) {
      markerDots.forEach((dot, index) => {
        // Draw outer circle with glow effect
        ctx.fillStyle = "rgba(239, 68, 68, 0.3)"; // Red with transparency
        ctx.beginPath();
        ctx.arc(dot.x, dot.y, 15, 0, 2 * Math.PI);
        ctx.fill();

        // Draw inner circle
        ctx.fillStyle = "#ef4444"; // Red
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(dot.x, dot.y, 8, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();

        // Draw X mark inside
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        const size = 4;
        ctx.beginPath();
        ctx.moveTo(dot.x - size, dot.y - size);
        ctx.lineTo(dot.x + size, dot.y + size);
        ctx.moveTo(dot.x + size, dot.y - size);
        ctx.lineTo(dot.x - size, dot.y + size);
        ctx.stroke();

        // Draw label
        ctx.fillStyle = "#ef4444";
        ctx.strokeStyle = "#000000";
        ctx.lineWidth = 3;
        ctx.font = "bold 11px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        const label = `M${index + 1}`;
        ctx.strokeText(label, dot.x, dot.y - 22);
        ctx.fillText(label, dot.x, dot.y - 22);
      });

      // Draw marker dots count badge
      if (markerDots.length > 0) {
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.fillRect(10, 50, 150, 30);
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 2;
        ctx.strokeRect(10, 50, 150, 30);

        ctx.fillStyle = "#ef4444";
        ctx.font = "bold 12px sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(`${markerDots.length} Table Marker${markerDots.length !== 1 ? 's' : ''}`, 18, 65);
      }
    }

    // Draw crosshair guidelines from mouse cursor
    if (mousePos) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.5)"; // Semi-transparent white
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]); // Dashed line

      // Vertical line
      ctx.beginPath();
      ctx.moveTo(mousePos.x, 0);
      ctx.lineTo(mousePos.x, height);
      ctx.stroke();

      // Horizontal line
      ctx.beginPath();
      ctx.moveTo(0, mousePos.y);
      ctx.lineTo(width, mousePos.y);
      ctx.stroke();

      ctx.setLineDash([]); // Reset dash pattern

      // Draw crosshair center point
      ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
      ctx.beginPath();
      ctx.arc(mousePos.x, mousePos.y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Request next frame
    animationFrameRef.current = requestAnimationFrame(drawFrame);
  }, [points, markerDots, width, height, overlayVisible, savedCalibrationData, selectionMode, mousePos]);

  // Start animation loop
  useEffect(() => {
    animationFrameRef.current = requestAnimationFrame(drawFrame);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [drawFrame]);

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    onPointSelect(x, y);
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    setMousePos({ x, y });
  };

  const handleMouseLeave = () => {
    setMousePos(null);
  };

  return (
    <div ref={containerRef} className="relative">
      {/* MJPEG stream image (hidden, used as source for canvas) */}
      <img
        ref={imgRef}
        src={videoStreamUrl}
        alt="Camera stream"
        style={{ display: "none" }}
        onLoad={() => {
          setStreamConnected(true);
          setStreamError(null);
        }}
        onError={() => {
          setStreamError("Failed to load video stream");
          setStreamConnected(false);
        }}
      />

      {/* Canvas for overlay */}
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        onClick={handleCanvasClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className="border border-gray-300 dark:border-gray-600 rounded-lg cursor-crosshair bg-gray-900"
      />

      {/* Connection status indicator */}
      {!streamConnected && (
        <div className="absolute top-2 left-2 px-2 py-1 bg-yellow-600 text-white text-xs rounded shadow">
          Connecting to camera stream...
        </div>
      )}

      {streamError && (
        <div className="absolute top-2 left-2 px-2 py-1 bg-red-600 text-white text-xs rounded shadow">
          {streamError}
        </div>
      )}
    </div>
  );
};

// Playing Area Calibration Step - Single step for table detection
const PlayingAreaCalibrationStep: React.FC<{
  onComplete: () => void;
}> = ({ onComplete }) => {
  const [points, setPoints] = useState<Point[]>([]);
  const [markerDots, setMarkerDots] = useState<MarkerDot[]>([]);
  const [selectionMode, setSelectionMode] = useState<'corners' | 'markers'>('corners');
  const [isApplying, setIsApplying] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [existingCalibration, setExistingCalibration] = useState<any>(null);
  const [savedCalibrationData, setSavedCalibrationData] = useState<CalibrationData | null>(null);

  const pointInstructions = [
    "Top-left corner of the playing area",
    "Top-right corner of the playing area",
    "Bottom-right corner of the playing area",
    "Bottom-left corner of the playing area",
  ];

  // Load existing calibration on mount
  useEffect(() => {
    const loadExistingCalibration = async () => {
      try {
        // Fetch from playing area config
        const response = await apiClient.getPlayingArea();
        if (response.success && response.data) {
          setExistingCalibration(response.data);

          // If corners exist, load them into points
          if (response.data.corners && response.data.corners.length === 4) {
            const loadedPoints: Point[] = response.data.corners.map(
              (corner: { x: number; y: number }, index: number) => ({
                x: corner.x,
                y: corner.y,
                id: `corner_${index}`,
                screenX: corner.x,
                screenY: corner.y,
                worldX: 0,
                worldY: 0,
              })
            );
            setPoints(loadedPoints);
          }

          // If marker dots exist, load them
          if (response.data.marker_dots && Array.isArray(response.data.marker_dots)) {
            const loadedMarkerDots: MarkerDot[] = response.data.marker_dots.map(
              (dot: { x: number; y: number }, index: number) => ({
                id: `marker_${index}_${Date.now()}`,
                x: dot.x,
                y: dot.y,
              })
            );
            setMarkerDots(loadedMarkerDots);
          }
        }

        // Also fetch calibration data for overlay
        const calibrationResponse = await apiClient.getCalibrationData();
        if (calibrationResponse.success && calibrationResponse.data) {
          const apiData = calibrationResponse.data;

          // Convert API calibration data to CalibrationData format
          const calibrationData: CalibrationData = {
            corners: (apiData.corners || []).map((corner: any, index: number) => ({
              id: `corner-${index}`,
              screenPosition: {
                x: Array.isArray(corner) ? corner[0] : corner.x,
                y: Array.isArray(corner) ? corner[1] : corner.y,
              },
              worldPosition: {
                x: Array.isArray(corner) ? corner[0] : corner.x,
                y: Array.isArray(corner) ? corner[1] : corner.y,
              },
              timestamp: Date.now(),
              confidence: apiData.accuracy || 1.0,
            })),
            transformationMatrix: apiData.transformation_matrix,
            calibratedAt: apiData.calibrated_at ? new Date(apiData.calibrated_at).getTime() : Date.now(),
            accuracy: apiData.accuracy,
            isValid: apiData.is_valid !== false && (apiData.corners || []).length >= 4,
            tableType: apiData.table_type,
            dimensions: apiData.dimensions,
          };

          setSavedCalibrationData(calibrationData);
        }
      } catch (error) {
        console.error("Failed to load existing calibration:", error);
      } finally {
        setIsLoading(false);
      }
    };

    loadExistingCalibration();
  }, []);

  const handlePointSelect = (screenX: number, screenY: number) => {
    if (selectionMode === 'markers') {
      // Add a new marker dot
      const newMarkerDot: MarkerDot = {
        id: `marker_${markerDots.length}_${Date.now()}`,
        x: screenX,
        y: screenY,
      };
      setMarkerDots([...markerDots, newMarkerDot]);
      return;
    }

    // Corner selection mode
    // Find the nearest corner and update it
    if (points.length === 0) {
      // No points yet, just add sequentially
      const newPoint: Point = {
        x: screenX,
        y: screenY,
        id: `corner_${points.length}`,
        screenX,
        screenY,
        worldX: 0,
        worldY: 0,
      };
      setPoints([newPoint]);
      return;
    }

    if (points.length < 4) {
      // Add next point
      const newPoint: Point = {
        x: screenX,
        y: screenY,
        id: `corner_${points.length}`,
        screenX,
        screenY,
        worldX: 0,
        worldY: 0,
      };
      setPoints([...points, newPoint]);
      return;
    }

    // Find nearest corner to update
    let nearestIndex = 0;
    let minDistance = Number.MAX_VALUE;

    points.forEach((point, index) => {
      const distance = Math.sqrt(
        Math.pow(point.screenX - screenX, 2) +
          Math.pow(point.screenY - screenY, 2)
      );
      if (distance < minDistance) {
        minDistance = distance;
        nearestIndex = index;
      }
    });

    // Update the nearest corner
    const newPoints = [...points];
    newPoints[nearestIndex] = {
      ...newPoints[nearestIndex],
      screenX,
      screenY,
      x: screenX,
      y: screenY,
    };
    setPoints(newPoints);
  };

  const removeLastPoint = () => {
    if (points.length > 0) {
      setPoints(points.slice(0, -1));
    }
  };

  const clearAllPoints = () => {
    setPoints([]);
  };

  const removeLastMarkerDot = () => {
    if (markerDots.length > 0) {
      setMarkerDots(markerDots.slice(0, -1));
    }
  };

  const clearAllMarkerDots = () => {
    setMarkerDots([]);
  };

  const applyCalibration = async () => {
    if (points.length !== 4) {
      alert("Please set all 4 corners of the playing area");
      return;
    }

    setIsApplying(true);
    try {
      // Report the calibration canvas dimensions; backend scales to physical resolution.
      const calibration_resolution_width = CALIBRATION_CANVAS_WIDTH;
      const calibration_resolution_height = CALIBRATION_CANVAS_HEIGHT;

      console.log(`Saving calibration with resolution: ${calibration_resolution_width}x${calibration_resolution_height}`);

      // Send the corners, marker dots, and resolution to the backend
      const response = await apiClient.updatePlayingArea({
        corners: points.map((p) => ({
          x: p.screenX,
          y: p.screenY,
        })),
        marker_dots: markerDots.map((dot) => ({
          x: dot.x,
          y: dot.y,
        })),
        calibration_resolution_width,
        calibration_resolution_height,
      });

      if (!response.success) {
        throw new Error(
          response.error || "Failed to apply playing area calibration"
        );
      }

      alert("Playing area calibration applied successfully!");
      onComplete();
    } catch (error) {
      console.error("Failed to apply calibration:", error);
      alert(
        "Failed to apply calibration: " +
          (error instanceof Error ? error.message : "Unknown error")
      );
    } finally {
      setIsApplying(false);
    }
  };

  const canProceed = points.length === 4;

  if (isLoading) {
    return (
      <div className="flex justify-center items-center py-12">
        <div className="text-gray-600 dark:text-gray-400">
          Loading calibration...
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
          Playing Area Calibration
        </h3>
        <p className="text-gray-600 dark:text-gray-400">
          Identify the 4 corners of the playing surface for accurate trajectory
          calculations
        </p>
        {existingCalibration && existingCalibration.calibrated && (
          <div className="mt-2 inline-flex items-center gap-2 px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300 rounded-full text-sm">
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                clipRule="evenodd"
              />
            </svg>
            Existing calibration loaded - adjust if needed
          </div>
        )}
      </div>

      <Card>
        <CardContent>
          <div className="flex justify-center">
            <VideoFeedCanvas
              onPointSelect={handlePointSelect}
              points={points}
              markerDots={markerDots}
              width={CALIBRATION_CANVAS_WIDTH}
              height={CALIBRATION_CANVAS_HEIGHT}
              overlayVisible={true}
              savedCalibrationData={savedCalibrationData}
              selectionMode={selectionMode}
            />
          </div>
        </CardContent>
      </Card>

      {/* Mode Selection */}
      <div className="flex justify-center gap-2">
        <Button
          variant={selectionMode === 'corners' ? 'primary' : 'outline'}
          onClick={() => setSelectionMode('corners')}
          className="flex-1"
        >
          Set Corners ({points.length}/4)
        </Button>
        <Button
          variant={selectionMode === 'markers' ? 'primary' : 'outline'}
          onClick={() => setSelectionMode('markers')}
          className="flex-1"
        >
          Mark Table Dots ({markerDots.length})
        </Button>
      </div>

      {/* Corner Controls */}
      {selectionMode === 'corners' && (
        <div className="grid grid-cols-2 gap-4">
          <Button
            variant="outline"
            onClick={removeLastPoint}
            disabled={points.length === 0}
          >
            Remove Last Corner
          </Button>
          <Button
            variant="outline"
            onClick={clearAllPoints}
            disabled={points.length === 0}
          >
            Clear All Corners
          </Button>
        </div>
      )}

      {/* Marker Dot Controls */}
      {selectionMode === 'markers' && (
        <div className="grid grid-cols-2 gap-4">
          <Button
            variant="outline"
            onClick={removeLastMarkerDot}
            disabled={markerDots.length === 0}
          >
            Remove Last Marker
          </Button>
          <Button
            variant="outline"
            onClick={clearAllMarkerDots}
            disabled={markerDots.length === 0}
          >
            Clear All Markers
          </Button>
        </div>
      )}

      <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
        <h4 className="font-medium text-gray-900 dark:text-white mb-2">
          Selected Corners:
        </h4>
        {points.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">
            No corners set. Click on the video to manually set the 4 corners of
            the playing area in order: top-left, top-right, bottom-right,
            bottom-left.
          </p>
        ) : (
          <div className="grid grid-cols-2 gap-4 text-sm">
            {points.map((point, index) => (
              <div key={point.id} className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">
                  {pointInstructions[index]}:
                </span>
                <span className="font-mono text-gray-800 dark:text-gray-200">
                  ({point.screenX.toFixed(0)}, {point.screenY.toFixed(0)})
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="flex justify-center">
        <Button
          onClick={applyCalibration}
          disabled={!canProceed || isApplying}
          variant={canProceed ? "primary" : "outline"}
          className="px-8"
        >
          {isApplying ? "Applying..." : "Apply Calibration"}
        </Button>
      </div>
    </div>
  );
};

// Main CalibrationWizard component
export const CalibrationWizard = observer(() => {
  const [completed, setCompleted] = useState(false);

  const handleComplete = () => {
    setCompleted(true);
  };

  const handleReset = () => {
    setCompleted(false);
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardContent>
          {!completed ? (
            <PlayingAreaCalibrationStep onComplete={handleComplete} />
          ) : (
            <div className="text-center py-12 space-y-4">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-green-100 dark:bg-green-900/30 rounded-full mb-4">
                <svg
                  className="w-10 h-10 text-green-600 dark:text-green-400"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
                Calibration Complete!
              </h3>
              <p className="text-gray-600 dark:text-gray-400 max-w-md mx-auto">
                Your playing area has been calibrated successfully. Trajectory
                calculations will now use the defined boundaries for accurate
                predictions.
              </p>
              <div className="flex justify-center gap-4 pt-4">
                <Button variant="outline" onClick={handleReset}>
                  Recalibrate
                </Button>
                <Button
                  variant="primary"
                  onClick={() => window.location.reload()}
                >
                  Done
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
});
