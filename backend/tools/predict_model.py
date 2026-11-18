"""Run one feature vector through the locally installed custom checkpoint."""
import argparse
import json

from backend.app.ml import FEATURES, registry

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--kind",choices=["gesture","eye"],required=True);p.add_argument("features",nargs="+",type=float);a=p.parse_args()
    registry.load()
    try: result=registry.predict(a.kind,a.features)
    except ValueError as exc: p.error(str(exc))
    print(json.dumps(result,indent=2))
