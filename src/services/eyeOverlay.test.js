import test from 'node:test';
import assert from 'node:assert/strict';
import {FACE_EYE_OVERLAYS,mapFaceLandmarkToOverlay} from './eyeOverlay.js';

test('overlay uses MediaPipe eyelid contours and separate iris landmark groups',()=>{
  assert.deepEqual(FACE_EYE_OVERLAYS.map(eye=>eye.contour[0]),[33,263]);
  assert.deepEqual(FACE_EYE_OVERLAYS.map(eye=>eye.iris),[[468,469,470,471,472],[473,474,475,476,477]]);
  assert.equal(FACE_EYE_OVERLAYS[0].color===FACE_EYE_OVERLAYS[1].color,false);
  const eyebrowIndices=new Set([46,53,52,65,55,70,63,105,66,107,276,283,282,295,285,300,293,334,296,336]);
  for(const eye of FACE_EYE_OVERLAYS) for(const index of eye.contour) assert.equal(eyebrowIndices.has(index),false);
});

test('overlay maps normalized points to mirrored object-fit cover coordinates',()=>{
  const point={x:.25,y:.5};
  assert.deepEqual(mapFaceLandmarkToOverlay(point,640,480,640,360),{x:480,y:180});
  const portrait=mapFaceLandmarkToOverlay(point,640,480,360,640);
  assert.ok(Math.abs(portrait.x-393.3333333333333)<1e-9);
  assert.equal(portrait.y,320);
  assert.deepEqual(mapFaceLandmarkToOverlay(point,640,480,640,360,false),{x:160,y:180});
});

test('overlay coordinate mapper rejects invalid dimensions',()=>{
  assert.throws(()=>mapFaceLandmarkToOverlay({x:0,y:0},0,480,640,360),/positive/);
  assert.throws(()=>mapFaceLandmarkToOverlay({x:NaN,y:0},640,480,640,360),/finite/);
});
