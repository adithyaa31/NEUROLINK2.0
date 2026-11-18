import numpy as np
import pytest

pd = pytest.importorskip("pandas", reason="install ml-engine/requirements.txt to test dataset tooling")
pytest.importorskip("sklearn", reason="install ml-engine/requirements.txt to test dataset tooling")

from backend.app.ml import GESTURES
from backend.tools.train_model import session_split, load_dataset
from backend.tools.validate_dataset import validate
from backend.tools.collect_webcam import append_collected_sample
import csv
import math
from datetime import datetime


def fixture_csv(path, missing_class=False):
    rows=[]
    for session in range(8):
        for label_id,label in enumerate(GESTURES):
            if missing_class and label==GESTURES[-1]: continue
            for sample in range(3):
                rows.append({"timestamp":f"2026-01-{session+1:02d}T00:00:00Z","session_id":f"session-{session}","label":label,**{f"f{i}":float((label_id+sample+i)%7)/7 for i in range(63)}})
    pd.DataFrame(rows).to_csv(path,index=False)


def test_session_split_is_disjoint_and_class_complete():
    y=np.tile(np.arange(len(GESTURES)),8)
    groups=np.repeat([f"session-{i}" for i in range(8)],len(GESTURES))
    x=np.zeros((len(y),63),dtype=np.float32)
    train,val,test=session_split(x,y,groups)
    assert not (set(groups[train])&set(groups[val]) or set(groups[train])&set(groups[test]) or set(groups[val])&set(groups[test]))
    assert all(set(np.unique(y[idx]))==set(range(len(GESTURES))) for idx in (train,val,test))


def test_two_session_split_keeps_second_session_held_out():
    # Split indices only; this is not written or used as training data.
    y=np.tile(np.arange(len(GESTURES)),32)
    groups=np.concatenate([np.repeat("session-1",96),np.repeat("session-2",96)])
    x=np.zeros((len(y),63),dtype=np.float32)
    train,val,test=session_split(x,y,groups,seed=17,train_session="session-1",test_session="session-2")
    assert set(groups[train])==set(groups[val])=={"session-1"}
    assert set(groups[test])=={"session-2"}
    assert not (set(groups[train]) & set(groups[test]))
    assert not (set(groups[val]) & set(groups[test]))
    assert all(set(np.unique(y[idx]))==set(range(len(GESTURES))) for idx in (train,val,test))
    assert len(val)==20  # 20% of 96, rounded by sklearn's test_size rule


def test_dataset_audit_and_load_reject_missing_class(tmp_path):
    path=tmp_path/"gesture.csv";fixture_csv(path,missing_class=True)
    report=validate(path,"gesture")
    assert not report["valid"] and "pointing" in report["missing_classes"]
    with pytest.raises(ValueError,match="missing"):
        load_dataset(path,"gesture")


def test_dataset_audit_reports_distribution_and_sessions(tmp_path):
    path=tmp_path/"gesture.csv";fixture_csv(path)
    report=validate(path,"gesture")
    assert report["valid"] and report["feature_dimensions"]==63
    assert report["session_count"]==8
    assert set(report["class_distribution"].values())=={24}
    assert len(load_dataset(path,"gesture")[1][0])==63


@pytest.mark.parametrize("mutation,expected",[("nan","non_numeric_or_nonfinite_feature_values"),("bad_label","unknown_labels"),("malformed","non_numeric_or_nonfinite_feature_values")])
def test_invalid_dataset_values_and_labels_are_reported(tmp_path,mutation,expected):
    path=tmp_path/"gesture.csv";fixture_csv(path)
    df=pd.read_csv(path)
    if mutation=="nan": df.loc[0,"f0"]=np.nan
    elif mutation=="bad_label": df.loc[0,"label"]="not_a_gesture"
    else:
        df["f0"]=df["f0"].astype(object)
        df.loc[0,"f0"]="malformed"
    df.to_csv(path,index=False)
    report=validate(path,"gesture")
    assert not report["valid"]
    assert report[expected]
    with pytest.raises(ValueError): load_dataset(path,"gesture")


def test_dataset_validation_reports_missing_columns(tmp_path):
    path=tmp_path/"gesture.csv";fixture_csv(path)
    df=pd.read_csv(path).drop(columns=["f62"]);df.to_csv(path,index=False)
    with pytest.raises(ValueError,match="missing"):
        validate(path,"gesture")


@pytest.mark.parametrize("kind,label,width",[("gesture","open_palm",63),("eye","closed",24)])
def test_collector_writes_header_timestamp_and_session(tmp_path,kind,label,width):
    path=tmp_path/kind/"samples.csv"
    sample=[0.25]*width
    written_at=append_collected_sample(path,"session-abc",label,sample,kind,"Left" if kind=="gesture" else "unknown")
    with path.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==1 and len(rows[0])==width+4
    assert rows[0]["session_id"]=="session-abc" and rows[0]["label"]==label
    assert rows[0]["timestamp"]==written_at and datetime.fromisoformat(written_at).tzinfo is not None
    assert math.isclose(float(rows[0]["f0"]),.25)
    append_collected_sample(path,"session-def",label,sample,kind)
    with path.open(newline="",encoding="utf-8") as f:
        assert len(list(csv.DictReader(f)))==2


def test_collector_rejects_invalid_feature_vectors(tmp_path):
    with pytest.raises(ValueError): append_collected_sample(tmp_path/"x.csv","s","open_palm",[0]*24,"gesture")
    with pytest.raises(ValueError): append_collected_sample(tmp_path/"x.csv","s","unknown",[0]*63,"gesture")
    with pytest.raises(ValueError): append_collected_sample(tmp_path/"x.csv","s","open_palm",[float("nan")]*63,"gesture")
