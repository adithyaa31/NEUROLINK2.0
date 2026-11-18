export const GESTURE_FEATURE_COUNT = 63;
export const EYE_FEATURE_COUNT = 24;
export const EYE_CONTOURS = [[33,160,158,133,153,144],[362,385,387,263,373,380]];

// Same outer-corner origin and outer-to-inner eye width scaling as backend.
export function normalizeEyeLandmarks(faceLandmarks) {
  if (!Array.isArray(faceLandmarks) || faceLandmarks.length <= 387) throw new TypeError('Expected Face Mesh landmarks containing both eyes.');
  const output=[];
  for (const indices of EYE_CONTOURS) {
    const eye=indices.map(i=>faceLandmarks[i]);
    if (eye.some(p=>!p||!Number.isFinite(p.x)||!Number.isFinite(p.y))) throw new TypeError('Eye landmarks must contain finite x and y values.');
    const dx=eye[3].x-eye[0].x,dy=eye[3].y-eye[0].y,scale=Math.hypot(dx,dy);
    if (scale<=1e-8) throw new TypeError('Eye contour is degenerate.');
    for (const p of eye) output.push((p.x-eye[0].x)/scale,(p.y-eye[0].y)/scale);
  }
  return output;
}

export function updateBlinkState(state, eyeState, now, {minClosedMs=50,maxClosedMs=500,cooldownMs=800}={}) {
  if (!['open','closed'].includes(eyeState)) throw new TypeError('Eye state must be open or closed.');
  if (eyeState==='closed') return {state:'closed',closedAt:state.state==='closed'?state.closedAt:now,lastEventAt:state.lastEventAt,event:false};
  const duration=state.state==='closed'&&state.closedAt!=null?now-state.closedAt:-1;
  const event=duration>=minClosedMs&&duration<=maxClosedMs&&now-(state.lastEventAt??-Infinity)>=cooldownMs;
  return {state:'open',closedAt:null,lastEventAt:event?now:state.lastEventAt,event};
}

export function smoothEyePrediction(history, prediction, confidence, {threshold=.65, windowSize=5, required=3}={}) {
  if (!['open','closed'].includes(prediction) || !Number.isFinite(confidence) || confidence < threshold) {
    history.length=0; return null;
  }
  history.push({prediction,confidence});
  if (history.length>windowSize) history.shift();
  const matching=history.filter(item=>item.prediction===prediction);
  if(matching.length<required) return null;
  return {label:prediction,confidence:matching.reduce((sum,item)=>sum+item.confidence,0)/matching.length};
}

export function standardizeEyeFeatures(features, mean, scale) {
  if(features.length!==24||mean.length!==24||scale.length!==24||scale.some(v=>!Number.isFinite(v)||v<=0)) throw new TypeError('Expected 24 eye features and matching valid preprocessing vectors.');
  return features.map((value,index)=>(value-mean[index])/scale[index]);
}

// Mirrors backend.app.features.normalize_landmarks exactly: wrist origin,
// canonical right-hand orientation, then max-absolute spatial scaling.
export function normalizeHandLandmarks(landmarks, handedness = 'Right') {
  if (!Array.isArray(landmarks) || landmarks.length !== 21 || landmarks.some(p => !Number.isFinite(p.x) || !Number.isFinite(p.y) || !Number.isFinite(p.z))) {
    throw new TypeError('Expected 21 finite hand landmarks with x, y, z coordinates.');
  }
  const wrist = landmarks[0];
  const points = landmarks.map(({x,y,z}) => [handedness.toLowerCase() === 'left' ? -(x-wrist.x) : x-wrist.x, y-wrist.y, z-wrist.z]);
  const scale = Math.max(...points.flat().map(Math.abs));
  return points.flatMap(p => p.map(v => { const normalized=scale > 0 ? v / scale : 0; return normalized===0 ? 0 : normalized; }));
}

export function smoothCursor(previous, next, alpha = 0.35) {
  if (!previous) return {x:next.x,y:next.y};
  return {x:previous.x+(next.x-previous.x)*alpha,y:previous.y+(next.y-previous.y)*alpha};
}

export function mapCursorPosition(point, calibration = {x:.5,y:.5}, sensitivity = 1) {
  const clamp=value=>Math.min(1,Math.max(0,value));
  return {x:clamp(.5+(point.x-calibration.x)*sensitivity),y:clamp(.5+(point.y-calibration.y)*sensitivity)};
}

export function pinchPressed(distance, wasPinching, lastSelectionAt, now, threshold = 0.055, cooldownMs = 700) {
  const pinching = Number.isFinite(distance) && distance <= threshold;
  return {pinching, select:pinching && !wasPinching && now-lastSelectionAt >= cooldownMs};
}
