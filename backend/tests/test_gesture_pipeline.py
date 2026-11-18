import numpy as np
import pytest
import torch
from backend.app.features import normalize_landmarks, normalize_eye_landmarks
from backend.app.main import app, VisionInput
from backend.app.ml import FeatureClassifier, EyeFeatureClassifier, ModelRegistry, GESTURES, EYE_STATES, FEATURES
from backend.app.blink import BlinkEventDetector

def test_landmark_feature_shape_and_normalization():
    points=np.stack([np.arange(21),np.arange(21)*.5,np.arange(21)*-.25],axis=1)
    features=normalize_landmarks(points)
    assert len(features)==63 and max(abs(value) for value in features)==1
    assert features[:3]==[0,0,0]
    assert FEATURES['gesture']==21*3==len(features)
    assert GESTURES==['open_palm','fist','thumbs_up','peace','pinch','pointing']

def test_left_and_right_hands_share_canonical_features():
    points=np.random.default_rng(4).normal(size=(21,3)).astype(np.float32)
    mirrored=points.copy(); mirrored[:,0]*=-1
    assert normalize_landmarks(points,'Right')==normalize_landmarks(mirrored,'Left')

@pytest.mark.parametrize('points', [np.zeros((20,3)), np.full((21,3),np.nan)])
def test_invalid_landmarks_are_rejected(points):
    with pytest.raises(ValueError): normalize_landmarks(points)

def test_eye_landmarks_have_24_scale_normalized_features():
    points=np.zeros((478,3),dtype=np.float32)
    for indices, offset in [((33,160,158,133,153,144),0),((362,385,387,263,373,380),1)]:
        points[list(indices),:2]=[[offset,0],[offset+.2,.1],[offset+.4,.1],[offset+1,0],[offset+.4,-.1],[offset+.2,-.1]]
    features=normalize_eye_landmarks(points)
    assert len(features)==24 and all(np.isfinite(features))
    assert FEATURES['eye']==12*2==len(features)
    assert EYE_STATES==['open','closed']
    with pytest.raises(ValueError): normalize_eye_landmarks(np.zeros((100,3)))

def test_eye_classifier_and_prediction_contract():
    model=EyeFeatureClassifier().eval()
    assert model(torch.zeros(1,24)).shape==(1,2)
    registry=ModelRegistry()
    assert registry.predict('eye',[0.0]*24)['status']=='model_not_trained'
    with pytest.raises(ValueError): registry.predict('eye',[0.0]*63)

def test_blink_detector_requires_completed_debounced_sequence():
    detector=BlinkEventDetector(cooldown_ms=800)
    assert not detector.update('closed',100)
    assert detector.update('open',200)
    detector.update('closed',300); assert not detector.update('open',330)
    detector.update('closed',500); assert not detector.update('open',600)  # cooldown
    detector.update('closed',700); assert not detector.update('open',1301)  # too long
    detector.update('closed',1400); assert detector.update('open',1500)
    detector.update('closed',1550); assert not detector.update('open',1650)  # cooldown from second event

def test_mediapipe_tasks_landmark_extractors_are_available():
    import mediapipe as mp
    from backend.tools.collect_webcam import build_landmarker_options
    assert mp.tasks.vision.HandLandmarker is not None
    assert mp.tasks.vision.FaceLandmarker is not None
    hand_type, hand_options=build_landmarker_options('gesture','not-downloaded.task')
    face_type, face_options=build_landmarker_options('eye','not-downloaded.task')
    assert hand_type is mp.tasks.vision.HandLandmarker and hand_options.num_hands==1
    assert face_type is mp.tasks.vision.FaceLandmarker and face_options.num_faces==1

def test_model_unavailable_is_explicit_and_invalid_features_still_fail():
    registry=ModelRegistry(); registry.models={'gesture':None,'eye':None}
    result=registry.predict('gesture',[0.0]*63)
    assert result=={'available':False,'status':'model_not_trained','prediction':None,'confidence':None,'latency_ms':None}
    with pytest.raises(ValueError): registry.predict('gesture',[0.0]*62)
    with pytest.raises(ValueError): registry.predict('gesture',[float('nan')]*63)

def test_corrupt_checkpoint_is_reported_without_crashing(tmp_path,monkeypatch):
    import backend.app.ml as ml
    monkeypatch.setattr(ml,'ROOT',tmp_path/'project')
    monkeypatch.setattr(ml,'MODEL_DIR',tmp_path)
    (tmp_path/'eye_model.pth').write_bytes(b'not a torch checkpoint')
    registry=ModelRegistry();registry.load()
    assert registry.status()['eye']['status']=='model_error'
    assert registry.predict('eye',[0.0]*24)['status']=='model_error'

def test_absent_checkpoints_cannot_be_marked_ready(tmp_path,monkeypatch):
    import backend.app.ml as ml
    monkeypatch.setattr(ml,'ROOT',tmp_path/'project')
    monkeypatch.setattr(ml,'MODEL_DIR',tmp_path)
    registry=ModelRegistry();registry.load()
    assert all(not model['available'] and model['status']=='model_not_trained' for model in registry.status().values())

def test_checkpoint_without_evaluation_sidecar_is_not_ready(tmp_path,monkeypatch):
    import backend.app.ml as ml
    monkeypatch.setattr(ml,'ROOT',tmp_path/'project')
    monkeypatch.setattr(ml,'MODEL_DIR',tmp_path)
    (tmp_path/'gesture_model.pth').touch()
    monkeypatch.setattr(ml.torch,'load',lambda *args,**kwargs:{'state_dict':{}})
    registry=ModelRegistry();registry.load()
    assert registry.status()['gesture']['status']=='model_error'
    assert 'Evaluation metadata is required' in registry.status()['gesture']['error']

def test_websocket_route_and_message_contract():
    assert any(getattr(route,'path',None)=='/ws/vision' for route in app.routes)
    body=VisionInput.model_validate({'kind':'gesture','features':[0.0]*63,'session_id':'test'})
    assert body.kind=='gesture' and len(body.features)==63
    with pytest.raises(ValueError): VisionInput.model_validate({'kind':'gesture','features':[0.0]*62})
