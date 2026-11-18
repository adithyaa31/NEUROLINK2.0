import {useEffect,useRef,useState} from 'react';
import {Eye,Wifi,WifiOff} from 'lucide-react';
import {FaceLandmarker,FilesetResolver} from '@mediapipe/tasks-vision';
import {api,apiBase} from '../services/api';
import {EYE_CONTOURS,normalizeEyeLandmarks,updateBlinkState,smoothEyePrediction} from '../services/gestureFeatures';
import {FACE_EYE_OVERLAYS,mapFaceLandmarkToOverlay} from '../services/eyeOverlay';

const TASK='https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const log=(label,session_id,metadata={})=>api('/api/events',{method:'POST',body:JSON.stringify({kind:'eye',label,session_id,metadata})}).catch(()=>{});
function Badge({children,tone=''}){return <span className={'badge '+tone}><i/>{children}</span>}

export default function EyeRecognition({status,blinkConfig}){
  const videoRef=useRef(null),canvasRef=useRef(null),streamRef=useRef(null),tracker=useRef(null),socket=useRef(null),raf=useRef(0),reconnect=useRef(0),reconnectAttempt=useRef(0),stopping=useRef(false),startToken=useRef(0),lastFrame=useRef(0),seq=useRef(0),lastHandled=useRef(0),blink=useRef({state:'unknown',closedAt:null,lastEventAt:-Infinity}),history=useRef([]),session=useRef(globalThis.crypto?.randomUUID?.()||`eye-${Date.now()}`);
  const [stream,setStream]=useState(null),[error,setError]=useState(''),[face,setFace]=useState(false),[socketState,setSocketState]=useState('idle'),[prediction,setPrediction]=useState(null),[latency,setLatency]=useState(null),[blinkCount,setBlinkCount]=useState(0),[blinkState,setBlinkState]=useState('unknown'),[select,setSelect]=useState('No completed blink');
  const available=Boolean(status?.models?.eye?.available),blinkConfigRef=useRef(blinkConfig);
  useEffect(()=>{blinkConfigRef.current=blinkConfig},[blinkConfig]);
  const cleanup=()=>{stopping.current=true;startToken.current++;clearTimeout(reconnect.current);cancelAnimationFrame(raf.current);socket.current?.close(1000,'camera stopped');socket.current=null;tracker.current?.close();tracker.current=null;streamRef.current?.getTracks().forEach(t=>{t.onended=null;t.stop()});streamRef.current=null;if(videoRef.current)videoRef.current.srcObject=null};
  useEffect(()=>()=>cleanup(),[]);
  const stop=()=>{cleanup();setStream(null);setFace(false);setPrediction(null);setLatency(null);setSocketState('idle');setBlinkState('unknown');history.current=[];blink.current={state:'unknown',closedAt:null,lastEventAt:-Infinity};log('camera_stopped',session.current)};
  const connectSocket=()=>{
    if(stopping.current||!streamRef.current||socket.current?.readyState===WebSocket.OPEN||socket.current?.readyState===WebSocket.CONNECTING)return;
    const url=apiBase.replace(/^http/,'ws')+'/ws/vision';
    let ws;try{ws=new WebSocket(url)}catch(e){setSocketState('error');setError(`Could not open vision WebSocket at ${url}: ${e.message}`);reconnect.current=setTimeout(connectSocket,1200);return}
    socket.current=ws;setSocketState('connecting');
    ws.onopen=()=>{if(socket.current!==ws||stopping.current)return;reconnectAttempt.current=0;setSocketState('connected');setError('')};
    ws.onerror=()=>{setSocketState('error');setError(`Vision WebSocket failed at ${url}. Check that FastAPI is running and reachable; retrying while the camera is active.`)};
    ws.onclose=event=>{if(socket.current===ws)socket.current=null;setPrediction(null);setLatency(null);if(!stopping.current&&streamRef.current){setSocketState('reconnecting');const delay=Math.min(1000*2**Math.min(reconnectAttempt.current++,4),10000);setError(`Vision WebSocket closed (code ${event.code}${event.reason?`, ${event.reason}`:''}). Reconnecting in ${Math.ceil(delay/1000)}s.`);reconnect.current=setTimeout(connectSocket,delay)}else setSocketState('idle')};
    ws.onmessage=event=>{try{const msg=JSON.parse(event.data);if(msg.status==='error'){setError(msg.message);return}if(msg.frame_id!==lastHandled.current)return;if(msg.available){const stable=smoothEyePrediction(history.current,msg.prediction,msg.confidence);if(!stable){blink.current={state:'unknown',closedAt:null,lastEventAt:blink.current.lastEventAt};setBlinkState('unknown');setPrediction(null);return}setPrediction(stable);setLatency(msg.latency_ms);const now=performance.now(),closedAt=blink.current.closedAt;const transition=updateBlinkState(blink.current,stable.label,now,blinkConfigRef.current);blink.current=transition;setBlinkState(transition.state);if(transition.event){setBlinkCount(n=>n+1);const active=document.activeElement,selectable=active?.matches?.('button,a,[role="button"],input,select')&&document.querySelector('.app-shell')?.contains(active);if(selectable){active.click();setSelect(`Selected ${active.getAttribute('aria-label')||active.textContent?.trim()||active.tagName}`)}else setSelect('Blink detected · focus an in-app control to select');log('blink_selection',session.current,{duration_ms:now-closedAt,selection_triggered:Boolean(selectable)});}else if(stable.label==='closed')setSelect('Eyes closed · waiting to reopen');else if(stable.label==='open')setSelect('No completed blink');}}catch{setError('Could not read the eye inference response.')}};
  };
  const start=async()=>{
    setError('');if(!navigator.mediaDevices?.getUserMedia){setError('Camera access is unavailable in this browser.');return}
    stopping.current=false;const token=++startToken.current;let acquired;
    try{
      acquired=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:640},height:{ideal:480}},audio:false});
      if(token!==startToken.current){acquired.getTracks().forEach(t=>t.stop());return}
      streamRef.current=acquired;setStream(acquired);videoRef.current.srcObject=acquired;await videoRef.current.play();
      if(token!==startToken.current)return;
      const vision=await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.3/wasm');
      tracker.current=await FaceLandmarker.createFromOptions(vision,{baseOptions:{modelAssetPath:TASK},runningMode:'VIDEO',numFaces:1,outputFaceBlendshapes:false});
      if(token!==startToken.current){tracker.current.close();tracker.current=null;return}
      acquired.getVideoTracks()[0].onended=()=>{setError('Camera disconnected.');stop()};
      stopping.current=false;connectSocket();
      const canvas=canvasRef.current,ctx=canvas.getContext('2d');
      const tick=now=>{raf.current=requestAnimationFrame(tick);const video=videoRef.current;if(!video||video.readyState<2||now-lastFrame.current<100)return;lastFrame.current=now;
        try{const result=tracker.current.detectForVideo(video,now),points=result.faceLandmarks?.[0];setFace(Boolean(points));const w=video.videoWidth||0,h=video.videoHeight||0,rect=canvas.getBoundingClientRect(),cw=Math.max(1,rect.width),ch=Math.max(1,rect.height),dpr=window.devicePixelRatio||1,pixelWidth=Math.max(1,Math.round(cw*dpr)),pixelHeight=Math.max(1,Math.round(ch*dpr));if(canvas.width!==pixelWidth||canvas.height!==pixelHeight){canvas.width=pixelWidth;canvas.height=pixelHeight}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,cw,ch);
          if(!points){setPrediction(null);blink.current={state:'unknown',closedAt:null,lastEventAt:blink.current.lastEventAt};setBlinkState('unknown');setSelect('Waiting for face');return}
          if(points.length<468||!w||!h)throw new Error(`Expected at least 468 face landmarks and valid video dimensions; received ${points.length}.`);
          for(const eye of FACE_EYE_OVERLAYS){ctx.fillStyle=eye.color;for(const index of eye.contour){const p=points[index];if(!p)continue;const position=mapFaceLandmarkToOverlay(p,w,h,cw,ch,false);ctx.beginPath();ctx.arc(position.x,position.y,2.2,0,Math.PI*2);ctx.fill()}if(points.length>Math.max(...eye.iris)){const center=mapFaceLandmarkToOverlay(points[eye.iris[0]],w,h,cw,ch,false);ctx.beginPath();ctx.arc(center.x,center.y,5,0,Math.PI*2);ctx.strokeStyle=eye.color;ctx.lineWidth=1.5;ctx.stroke();ctx.fillStyle='#F4F7FF';for(const index of eye.iris.slice(1)){const p=points[index];const position=mapFaceLandmarkToOverlay(p,w,h,cw,ch,false);ctx.beginPath();ctx.arc(position.x,position.y,1.8,0,Math.PI*2);ctx.fill()}}}
          if(available&&socket.current?.readyState===WebSocket.OPEN){const frame_id=++seq.current;socket.current.send(JSON.stringify({kind:'eye',features:normalizeEyeLandmarks(points),session_id:session.current,frame_id}));lastHandled.current=frame_id;}
        }catch(e){setError(`Eye landmark processing failed: ${e.message}`)}
      };raf.current=requestAnimationFrame(tick);
    }catch(e){if(token!==startToken.current)return;socket.current?.close();socket.current=null;tracker.current?.close();tracker.current=null;acquired?.getTracks().forEach(t=>t.stop());streamRef.current=null;if(videoRef.current)videoRef.current.srcObject=null;setStream(null);setError(e.name==='NotAllowedError'?'Camera permission was denied.':`Could not start eye tracking: ${e.message}`)}
  };
  const modelState=status?.models?.eye?.status;
  return <section className="card camera-card"><div className="camera-header"><div><span className="micro">LIVE EYE INPUT</span><h3>Camera preview</h3></div><Badge tone={available?'good':'warn'}>{available?'MODEL READY':modelState==='model_error'?'MODEL ERROR':modelState==='model_loading'?'MODEL LOADING':'MODEL NOT TRAINED'}</Badge></div><div className="camera-view"><video ref={videoRef} autoPlay playsInline muted style={{display:stream?'block':'none'}}/><canvas ref={canvasRef} style={{position:'absolute',inset:0,width:'100%',height:'100%',pointerEvents:'none',transform:'scaleX(-1)',display:stream?'block':'none'}}/>{!stream&&<div className="camera-placeholder"><div className="camera-glyph"><Eye/></div><b>Camera is off</b><small>Face landmarks are processed locally after camera starts.</small></div>}</div>{error&&<div className="error-msg" role="alert">{error}</div>}<div className="camera-controls">{stream?<button className="secondary-button" onClick={stop}>Stop camera</button>:<button className="primary-button" onClick={start}>Start camera</button>}<span className="camera-status">{face?'Face detected':stream?'Waiting for face':'Camera off'}</span></div><div className="data-line"><span>Eye state</span><b>{prediction?.label||'—'}</b></div><div className="data-line"><span>Confidence</span><b>{prediction?`${(prediction.confidence*100).toFixed(1)}%`:'—'}</b></div><div className="data-line"><span>Inference latency</span><b>{latency!=null?`${latency.toFixed(2)} ms`:'—'}</b></div><div className="data-line"><span>WebSocket</span><b>{!stream?<><WifiOff size={14}/> Idle · start camera to connect</>:socketState==='connected'?<><Wifi size={14}/> Connected</>:<><WifiOff size={14}/> {socketState}</>}</b></div><div className="data-line"><span>Backend</span><b>{status?.api==='online'?'Reachable':status?.api==='offline'?'Unavailable':'Checking…'}</b></div><div className="data-line"><span>Blink sequence</span><b>{blinkState.toUpperCase()}</b></div><div className="data-line"><span>Completed blink events</span><b>{blinkCount}</b></div><div className="data-line"><span>Selection</span><b>{select}</b></div></section>
}
