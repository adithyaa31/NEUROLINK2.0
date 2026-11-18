import {useCallback,useEffect,useRef,useState} from 'react';
import {FilesetResolver,HandLandmarker} from '@mediapipe/tasks-vision';
import {api,apiBase} from '../services/api';
import {normalizeHandLandmarks,pinchPressed,smoothCursor,mapCursorPosition} from '../services/gestureFeatures';
import {stablePrediction} from '../services/interactionCoordinator';

const VISION_VERSION='1.2.3';
const TASK_URL='https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const CONNECTIONS=[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const log=(kind,label,session_id,metadata={})=>api('/api/events',{method:'POST',body:JSON.stringify({kind,label,session_id,metadata})}).catch(error=>console.warn('Could not log gesture event:',error.message));

export function useGestureRecognition({videoRef,canvasRef,modelAvailable}) {
  const [stream,setStream]=useState(null),[cameraState,setCameraState]=useState('off'),[mediapipeState,setMediapipeState]=useState('idle'),[handDetected,setHandDetected]=useState(false),[landmarkCount,setLandmarkCount]=useState(0),[featureReady,setFeatureReady]=useState(false),[featureDiagnostics,setFeatureDiagnostics]=useState(null),[videoInfo,setVideoInfo]=useState(null),[socketState,setSocketState]=useState('disconnected'),[inferenceState,setInferenceState]=useState('idle'),[prediction,setPrediction]=useState(null),[latency,setLatency]=useState(null),[error,setError]=useState(''),[cursorEnabled,setCursorEnabled]=useState(false),[cursorSensitivity,setCursorSensitivity]=useState(1),[cursorSmoothing,setCursorSmoothing]=useState(.32),[pinchThreshold,setPinchThreshold]=useState(.055),[cursor,setCursor]=useState(null),[pinchActive,setPinchActive]=useState(false);
  const session=useRef(globalThis.crypto?.randomUUID?.()||`session-${Date.now()}`).current;
  const landmarker=useRef(null),socket=useRef(null),animation=useRef(null),lastFrame=useRef(0),lastSend=useRef(0),frameId=useRef(0),latestFrameId=useRef(0),wasHand=useRef(false),wasPinching=useRef(false),lastSelection=useRef(-Infinity),smoothed=useRef(null),reconnect=useRef(null),reconnectAttempt=useRef(0),stopping=useRef(false),cursorEnabledRef=useRef(false),modelAvailableRef=useRef(modelAvailable),streamRef=useRef(null),startToken=useRef(0),predictionHistory=useRef([]),featureDiagnosticsRef=useRef(null),cursorConfig=useRef({sensitivity:1,smoothing:.32,pinchThreshold:.055}),lastCursorPoint=useRef(null),calibration=useRef({x:.5,y:.5});
  useEffect(()=>{cursorEnabledRef.current=cursorEnabled},[cursorEnabled]);
  useEffect(()=>{modelAvailableRef.current=modelAvailable},[modelAvailable]);
  useEffect(()=>{cursorConfig.current={sensitivity:cursorSensitivity,smoothing:cursorSmoothing,pinchThreshold}},[cursorSensitivity,cursorSmoothing,pinchThreshold]);
  const calibrateCursor=useCallback(()=>{if(lastCursorPoint.current){calibration.current={...lastCursorPoint.current};smoothed.current=null;return true}return false},[]);
  const stop=useCallback((reason='user')=>{
    const cameraWasActive=Boolean(streamRef.current);
    stopping.current=true;startToken.current++;clearTimeout(reconnect.current);cancelAnimationFrame(animation.current);
    socket.current?.close(1000,'camera stopped');socket.current=null;setSocketState('disconnected');
    landmarker.current?.close();landmarker.current=null;
    streamRef.current?.getTracks().forEach(track=>{track.onended=null;track.stop()});streamRef.current=null;setStream(null);setCameraState('off');
    setHandDetected(false);setMediapipeState('idle');setLandmarkCount(0);setFeatureReady(false);setFeatureDiagnostics(null);setVideoInfo(null);setPrediction(null);setLatency(null);setInferenceState('idle');setPinchActive(false);setCursor(null);smoothed.current=null;wasHand.current=false;wasPinching.current=false;predictionHistory.current=[];
    if(cameraWasActive) log('camera',reason==='disconnect'?'disconnected':'stopped',session,{reason});
  },[session]);
  const connect=useCallback(()=>{
    if(stopping.current||!streamRef.current||socket.current?.readyState===WebSocket.OPEN||socket.current?.readyState===WebSocket.CONNECTING)return;
    const socketUrl=apiBase.replace(/^http/,'ws')+'/ws/vision';
    console.info('[NEUROLINK gesture] WebSocket connecting',socketUrl);
    let ws;try{ws=new WebSocket(socketUrl)}catch(e){setSocketState('error');setError(`WebSocket could not start: ${e.message}`);reconnect.current=setTimeout(connect,1200);return}
    socket.current=ws;setSocketState('connecting');
    ws.onopen=()=>{if(socket.current!==ws||stopping.current)return;reconnectAttempt.current=0;console.info('[NEUROLINK WEBSOCKET] Open handshake succeeded',socketUrl);setSocketState('open');log('websocket','connected',session,{url:socketUrl});setError('')};
    ws.onmessage=event=>{if(socket.current!==ws||stopping.current)return;try{const result=JSON.parse(event.data);if(result.status==='error'){console.error('[NEUROLINK BACKEND] Rejected vision frame',result);setInferenceState('failed');setError(result.message||'Gesture inference failed.');log('inference','failed',session,{message:result.message});return}if(result.frame_id!=null&&result.frame_id>latestFrameId.current)return;setSocketState('connected');if(result.status==='model_not_trained'||result.available===false){console.warn('[NEUROLINK INFERENCE] Backend reports model unavailable',result.status);setInferenceState('failed');predictionHistory.current=[];setPrediction(null);setLatency(null);return}if(result.available){if(!['open_palm','fist','thumbs_up','peace','pinch','pointing'].includes(result.prediction)||!Number.isFinite(result.confidence)||result.confidence<0||result.confidence>1||!Number.isFinite(result.latency_ms)){throw new Error('The model returned an invalid prediction response.')}setInferenceState('success');const stable=stablePrediction(predictionHistory.current,result.prediction,result.confidence,{threshold:.8,frames:3});setPrediction(stable?{label:stable,confidence:result.confidence}:null);setLatency(result.latency_ms);console.info('[NEUROLINK INFERENCE] Actual model response',{frame_id:result.frame_id,features:featureDiagnosticsRef.current?.normalizedCount,prediction:result.prediction,confidence:result.confidence,latency_ms:result.latency_ms})}}catch(e){setInferenceState('failed');setError(e.message||'Could not read the inference response.');console.error('[NEUROLINK WEBSOCKET] Response error:',e)}};
    ws.onerror=event=>{console.error('[NEUROLINK WEBSOCKET] Error',socketUrl,event);setSocketState('error');setError(`Vision WebSocket failed at ${socketUrl}; retrying while camera remains active.`)};
    ws.onclose=event=>{console.warn('[NEUROLINK WEBSOCKET] Closed',{code:event.code,reason:event.reason,clean:event.wasClean,url:socketUrl});if(socket.current===ws)socket.current=null;if(!stopping.current&&streamRef.current){setSocketState('reconnecting');const delay=Math.min(1000*2**reconnectAttempt.current++,10000);reconnect.current=setTimeout(connect,delay)}else setSocketState('disconnected');setPrediction(null);setLatency(null);log('websocket','disconnected',session,{code:event.code,reason:event.reason,clean:event.wasClean});};
  },[session]);
  const start=useCallback(async()=>{
    if(!navigator.mediaDevices?.getUserMedia){setError('Camera access is not supported in this browser. Use a current browser on localhost or HTTPS.');return}
    stopping.current=false;const token=++startToken.current;setError('');setCameraState('starting');setMediapipeState('loading');setInferenceState('idle');
    let acquired;
    try {
      acquired=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:640},height:{ideal:480}},audio:false});
      console.info('[NEUROLINK gesture] Camera stream acquired',acquired.getVideoTracks().map(track=>({label:track.label,readyState:track.readyState,settings:track.getSettings()})));
      streamRef.current=acquired;setStream(acquired);
      log('camera','started',session,{width:640,height:480});
      if(!videoRef.current)throw new Error('Camera preview could not be initialized.');
      videoRef.current.srcObject=acquired;await videoRef.current.play();
      setVideoInfo({width:videoRef.current.videoWidth,height:videoRef.current.videoHeight,paused:videoRef.current.paused});
      console.info('[NEUROLINK gesture] Video playing',{width:videoRef.current.videoWidth,height:videoRef.current.videoHeight,paused:videoRef.current.paused,readyState:videoRef.current.readyState});
      if(token!==startToken.current)return;
      setMediapipeState('loading');const vision=await FilesetResolver.forVisionTasks(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VISION_VERSION}/wasm`);
      landmarker.current=await HandLandmarker.createFromOptions(vision,{baseOptions:{modelAssetPath:TASK_URL},runningMode:'VIDEO',numHands:1});
      setMediapipeState('initialized');console.info('[NEUROLINK MEDIAPIPE] Hand Landmarker initialized');
      console.info('[NEUROLINK gesture] MediaPipe Hand Landmarker initialized');
      if(token!==startToken.current){landmarker.current?.close();landmarker.current=null;return}
      acquired.getVideoTracks()[0].onended=()=>{setError('Camera disconnected. Start the camera again when it is available.');stop('disconnect')};
      setCameraState('tracking');connect();
      const canvas=canvasRef.current,ctx=canvas?.getContext('2d');
      const tick=now=>{
        if(stopping.current||!streamRef.current)return;
        animation.current=requestAnimationFrame(tick);
        const video=videoRef.current;if(!video||video.readyState<2||now-lastFrame.current<100)return;lastFrame.current=now;
        try{
          if(!landmarker.current)throw new Error('Hand Landmarker is not initialized.');
          const result=landmarker.current.detectForVideo(video,now);const points=result.landmarks?.[0];
          if(canvas&&ctx){const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(1,Math.round(rect.width)),h=Math.max(1,Math.round(rect.height)),vw=video.videoWidth||0,vh=video.videoHeight||0;if(vw&&vh){if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);if(points?.length===21){const scale=Math.max(w/vw,h/vh),ox=(w-vw*scale)/2,oy=(h-vh*scale)/2;const xy=p=>[p.x*vw*scale+ox,p.y*vh*scale+oy];ctx.strokeStyle='#00E5FF';ctx.fillStyle='#F4F7FF';ctx.lineWidth=2;ctx.shadowColor='#00E5FF';ctx.shadowBlur=8;for(const [a,b] of CONNECTIONS){const [ax,ay]=xy(points[a]),[bx,by]=xy(points[b]);ctx.beginPath();ctx.moveTo(ax,ay);ctx.lineTo(bx,by);ctx.stroke()}for(const p of points){const [x,y]=xy(p);ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill()}}else if(points)console.error('[NEUROLINK RENDERING] Expected 21 landmarks',points.length)}}else if(points)console.error('[NEUROLINK RENDERING] Canvas context unavailable');
          if(!points){setLandmarkCount(0);setFeatureReady(false);setInferenceState('idle');if(wasHand.current){console.info('[NEUROLINK LANDMARKS] Hand lost');setPrediction(null);setLatency(null)}setHandDetected(false);wasHand.current=false;wasPinching.current=false;lastCursorPoint.current=null;smoothed.current=null;setPinchActive(false);setCursor(null);return}
          setLandmarkCount(points.length);
          if(points.length!==21){setFeatureReady(false);throw new Error(`MediaPipe returned ${points.length} hand landmarks; expected 21.`)}
          if(!wasHand.current){console.info('[NEUROLINK gesture] Hand detected: 21 landmarks');setHandDetected(true);log('hand','detected',session,{landmarks:points.length})}wasHand.current=true;
          const handedness=result.handednesses?.[0]?.[0]?.categoryName||'Right';
          const rawFeatures=points.flatMap(point=>[point.x,point.y,point.z]);
          const features=normalizeHandLandmarks(points,handedness);
          const validFeatures=points.length===21&&rawFeatures.length===63&&rawFeatures.every(Number.isFinite)&&features.length===63&&features.every(Number.isFinite);
          const diagnostics={landmarkCount:points.length,rawFeatureCount:rawFeatures.length,normalizedFeatureCount:features.length,valid:validFeatures,firstValues:features.slice(0,6),frameId:frameId.current+1};featureDiagnosticsRef.current=diagnostics;setFeatureDiagnostics(diagnostics);setFeatureReady(validFeatures);
          if(!validFeatures){console.error('[NEUROLINK FEATURES] Invalid vector',diagnostics);throw new Error(`Feature validation failed: landmarks=${points.length}, raw=${rawFeatures.length}, normalized=${features.length}.`)}
          if(frameId.current===0)console.info('[NEUROLINK FEATURES] Validated vector',{landmarks:diagnostics.landmarkCount,raw:diagnostics.rawFeatureCount,normalized:diagnostics.normalizedFeatureCount,valid:diagnostics.valid});
          if(modelAvailableRef.current&&socket.current?.readyState===WebSocket.OPEN&&now-lastSend.current>=100){try{const frame_id=++frameId.current;latestFrameId.current=frame_id;diagnostics.frameId=frame_id;featureDiagnosticsRef.current=diagnostics;setFeatureDiagnostics({...diagnostics});setInferenceState('processing');const payload={kind:'gesture',features,session_id:session,frame_id};if(payload.features.length!==63||payload.features.some(value=>!Number.isFinite(value)))throw new Error('Refusing to send invalid features.');socket.current.send(JSON.stringify(payload));lastSend.current=now;console.info('[NEUROLINK WEBSOCKET] Sent gesture features',{frame_id,feature_count:payload.features.length})}catch(e){setInferenceState('failed');setError(`Could not send landmarks to inference: ${e.message}`);console.error('[NEUROLINK FEATURES] Feature send failed',e)}}
          const thumb=points[4],index=points[8],d=Math.hypot(thumb.x-index.x,thumb.y-index.y);const pinch=pinchPressed(d,wasPinching.current,lastSelection.current,now,cursorConfig.current.pinchThreshold);wasPinching.current=pinch.pinching;setPinchActive(pinch.pinching);
          if(cursorEnabledRef.current){const raw={x:1-index.x,y:index.y};lastCursorPoint.current=raw;const calibrated=mapCursorPosition(raw,calibration.current,cursorConfig.current.sensitivity);smoothed.current=smoothCursor(smoothed.current,calibrated,cursorConfig.current.smoothing);const main=document.querySelector('.main'),rect=main?.getBoundingClientRect();if(rect){const pos={x:rect.left+smoothed.current.x*rect.width,y:Math.min(window.innerHeight-8,64+smoothed.current.y*(window.innerHeight-64))};setCursor(pos);if(pinch.select){lastSelection.current=now;const target=document.elementFromPoint(pos.x,pos.y)?.closest('button,a,[role="button"],input,select');if(target&&document.querySelector('.app-shell')?.contains(target)){target.click();log('selection','pinch',session,{x:Math.round(smoothed.current.x*100),y:Math.round(smoothed.current.y*100)});}}}}else setCursor(null);
          if(modelAvailableRef.current&&socket.current?.readyState!==WebSocket.OPEN&&now-lastSend.current>5000){setPrediction(null);setLatency(null)}
        }catch(e){setError(`Hand tracking failed: ${e.message}`);console.error('Gesture frame processing error:',e)}
      };
      animation.current=requestAnimationFrame(tick);
    }catch(e){if(acquired){acquired.getTracks().forEach(t=>{t.onended=null;t.stop()});log('camera','stopped',session,{reason:'initialization-failed'})}landmarker.current?.close();landmarker.current=null;streamRef.current=null;setStream(null);setCameraState('error');setMediapipeState('failed');setSocketState('disconnected');const msg=e.name==='NotAllowedError'?'Camera permission was denied. Allow access in browser settings and retry.':e.name==='NotFoundError'?'No camera was found. Connect a camera and retry.':`Could not start hand tracking: ${e.message||'check camera and network access'}`;setError(msg);console.error('[NEUROLINK CAMERA] Camera/MediaPipe initialization failed',e)}
  },[canvasRef,connect,session,stop,videoRef]);
  useEffect(()=>()=>{stop('unmount')},[stop]);
  return {stream,cameraState,mediapipeState,handDetected,landmarkCount,featureReady,featureDiagnostics,videoInfo,socketState,inferenceState,prediction,latency,error,cursorEnabled,setCursorEnabled,cursorSensitivity,setCursorSensitivity,cursorSmoothing,setCursorSmoothing,pinchThreshold,setPinchThreshold,calibrateCursor,pinchActive,cursor,start,stop,session};
}
