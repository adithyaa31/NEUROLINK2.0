// MediaPipe Face Landmarker (478 point output) eye topology. The feature
// extractor has its own stable 24-value contract in gestureFeatures.js.
export const FACE_EYE_OVERLAYS = [
  {
    name: 'right-eye',
    color: '#00E5FF',
    contour: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    iris: [468, 469, 470, 471, 472],
  },
  {
    name: 'left-eye',
    color: '#845BFF',
    contour: [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
    iris: [473, 474, 475, 476, 477],
  },
];

/**
 * Map a normalized Face Landmarker point into the visible, mirrored video box.
 * This matches CSS object-fit: cover and transform: scaleX(-1) exactly.
 */
export function mapFaceLandmarkToOverlay(point, videoWidth, videoHeight, viewWidth, viewHeight, mirrored = true) {
  if (![point?.x, point?.y, videoWidth, videoHeight, viewWidth, viewHeight].every(Number.isFinite) ||
      videoWidth <= 0 || videoHeight <= 0 || viewWidth <= 0 || viewHeight <= 0) {
    throw new TypeError('Expected finite landmark and positive video and overlay dimensions.');
  }
  const scale = Math.max(viewWidth / videoWidth, viewHeight / videoHeight);
  const renderedWidth = videoWidth * scale;
  const renderedHeight = videoHeight * scale;
  const offsetX = (viewWidth - renderedWidth) / 2;
  const offsetY = (viewHeight - renderedHeight) / 2;
  const normalizedX = mirrored ? 1 - point.x : point.x;
  return {
    x: offsetX + normalizedX * renderedWidth,
    y: offsetY + point.y * renderedHeight,
  };
}
