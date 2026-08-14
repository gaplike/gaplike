"""Run all PE jobs (scenario x model) across 2 worker processes."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import time
import numpy as np
from multiprocessing import Pool

MODELS = ["full", "diag", "bare", "psd"]
SCEN = ["A", "B"]
JOBS = [(s, "full") for s in SCEN] + [(s, m) for s in SCEN for m in MODELS if m != "full"]

_cache = {}


def _get_scenario(key):
    if key not in _cache:
        import scenarios as S
        _cache[key] = S.build_scenario(key, verbose=False)
    return _cache[key]


def job(args):
    skey, mk = args
    import pe
    t0 = time.time()
    sc = _get_scenario(skey)
    chain, acc = pe.run_pe(sc, mk, nwalkers=48, nburn=1800, nsteps=4200, thin=7,
                           seed=abs(hash((skey, mk))) % 2**31)
    med = np.median(chain, axis=0)
    return f"[{skey:12s} {mk:4s}] {time.time()-t0:6.1f}s acc={acc:.2f} med-truth={np.round(med - pe.X_TRUE, 4)}"


if __name__ == "__main__":
    t0 = time.time()
    with Pool(2) as p:
        for line in p.imap_unordered(job, JOBS):
            print(line, flush=True)
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)
