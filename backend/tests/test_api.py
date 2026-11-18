import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from backend.app.main import app, health, gesture_predict, eye_predict, EventInput, PredictInput
from backend.app.features import normalize_landmarks, eye_aspect_ratio

def test_health():
    assert health()['status']=='online'

def test_missing_model_is_honest():
    from backend.app.ml import registry
    registry.models={'gesture':None,'eye':None}
    result=gesture_predict(PredictInput(features=[0]*63))
    assert result['status']=='model_not_trained'

def test_event_validation():
    with pytest.raises(ValidationError): EventInput(kind='gesture',label='test',confidence=2)

def test_event_metadata_is_size_bounded():
    with pytest.raises(ValidationError): EventInput(kind='gesture',label='test',metadata={'payload':'x'*5000})

def test_event_indexes_and_explicit_pruning(tmp_path, monkeypatch):
    import backend.app.database as database
    from backend.app.database import initialize, log_event, delete_events_before, fetch_events
    monkeypatch.setattr(database, 'DATABASE_PATH', tmp_path/'events.db')
    initialize()
    log_event('gesture','old',metadata='{}')
    with database.connect() as db:
        db.execute("UPDATE events SET timestamp='2020-01-01 00:00:00'")
    log_event('gesture','new',metadata='{}')
    with database.connect() as db:
        indexes={row['name'] for row in db.execute("PRAGMA index_list(events)")}
    assert {'idx_events_kind_id','idx_events_timestamp'} <= indexes
    assert delete_events_before('2021-01-01 00:00:00') == 1
    assert [event['label'] for event in fetch_events()] == ['new']

def test_loopback_event_clear_endpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import backend.app.database as database
    from backend.app.main import clear_local_events
    monkeypatch.setattr(database, 'DATABASE_PATH', tmp_path/'clear.db')
    database.initialize(); database.log_event('gesture','test')
    response=clear_local_events(SimpleNamespace(client=SimpleNamespace(host='127.0.0.1')))
    assert response=={'deleted':1}
    assert database.fetch_events()==[]
    with pytest.raises(HTTPException) as error:
        clear_local_events(SimpleNamespace(client=SimpleNamespace(host='192.0.2.1')))
    assert error.value.status_code==403

def test_landmark_normalization():
    points=[[float(i),0,0] for i in range(21)]
    result=normalize_landmarks(points)
    assert len(result)==63 and max(abs(v) for v in result)==1

def test_eye_aspect_ratio():
    assert eye_aspect_ratio([[0,0],[1,1],[2,1],[4,0],[2,-1],[1,-1]]) > 0

def test_eye_endpoint_requires_24_features():
    with pytest.raises(HTTPException): eye_predict(PredictInput(features=[0]*63))
    assert eye_predict(PredictInput(features=[0]*24))['status']=='model_not_trained'
