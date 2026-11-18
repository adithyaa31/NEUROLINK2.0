import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeHandLandmarks,normalizeEyeLandmarks,updateBlinkState,pinchPressed,smoothCursor,mapCursorPosition} from './gestureFeatures.js';

const points=Array.from({length:21},(_,i)=>({x:i/20,y:(i%5)/5,z:i/100}));
test('normalizer returns 63 finite wrist-relative features',()=>{
  const features=normalizeHandLandmarks(points);
  assert.equal(features.length,63);assert.ok(features.every(Number.isFinite));
  assert.deepEqual(features.slice(0,3),[0,0,0]);assert.equal(Math.max(...features.map(Math.abs)),1);
});
test('left hand maps to canonical right-hand feature orientation',()=>{
  const mirrored=points.map(p=>({...p,x:-p.x}));
  assert.deepEqual(normalizeHandLandmarks(points,'Right'),normalizeHandLandmarks(mirrored,'Left'));
});
test('invalid landmarks are rejected',()=>{
  assert.throws(()=>normalizeHandLandmarks(points.slice(1)),/21/);
  assert.throws(()=>normalizeHandLandmarks(points.map((p,i)=>i? p:{...p,x:NaN})),/finite/);
});
test('hand extraction rejects non-finite and degenerate data without malformed vectors',()=>{
  assert.throws(()=>normalizeHandLandmarks(points.map((p,i)=>i===4?{...p,z:Infinity}:p)),/finite/);
  const collapsed=Array.from({length:21},()=>({x:1,y:1,z:1}));
  const features=normalizeHandLandmarks(collapsed);
  assert.equal(features.length,63);assert.ok(features.every(Number.isFinite));assert.ok(features.every(v=>v===0));
});
test('cursor smoothing initializes then interpolates',()=>{
  assert.deepEqual(smoothCursor(null,{x:1,y:0}),{x:1,y:0});
  assert.deepEqual(smoothCursor({x:0,y:0},{x:1,y:1},.25),{x:.25,y:.25});
});
test('cursor calibration and sensitivity map only to the in-app normalized viewport',()=>{
  assert.deepEqual(mapCursorPosition({x:.7,y:.3},{x:.5,y:.5},1),{x:.7,y:.3});
  const amplified=mapCursorPosition({x:.7,y:.3},{x:.5,y:.5},2);
  assert.ok(Math.abs(amplified.x-.9)<1e-12&&Math.abs(amplified.y-.1)<1e-12);
  assert.deepEqual(mapCursorPosition({x:1,y:0},{x:.5,y:.5},4),{x:1,y:0});
});
test('pinch selection fires on an edge and observes cooldown',()=>{
  assert.deepEqual(pinchPressed(.04,false,0,1000),{pinching:true,select:true});
  assert.deepEqual(pinchPressed(.04,true,1000,1200),{pinching:true,select:false});
  assert.deepEqual(pinchPressed(.04,false,1000,1200),{pinching:true,select:false});
  assert.deepEqual(pinchPressed(.2,true,1000,1300),{pinching:false,select:false});
});

test('eye landmark normalization produces the 24 value contract',()=>{
  const points=Array.from({length:478},()=>({x:0,y:0,z:0}));
  [[33,160,158,133,153,144],[362,385,387,263,373,380]].forEach((ids,n)=>ids.forEach((id,i)=>{const p=[[0,0],[.2,.1],[.4,.1],[1,0],[.4,-.1],[.2,-.1]][i];points[id]={x:p[0]+n,y:p[1],z:0}}));
  assert.equal(normalizeEyeLandmarks(points).length,24);
});

test('blink events require open-close-open duration and cooldown',()=>{
  let state={state:'unknown',closedAt:null,lastEventAt:-Infinity};
  state=updateBlinkState(state,'closed',100);assert.equal(state.event,false);
  state=updateBlinkState(state,'open',200);assert.equal(state.event,true);
  state=updateBlinkState(state,'closed',300);state=updateBlinkState(state,'open',600);assert.equal(state.event,false);
});
