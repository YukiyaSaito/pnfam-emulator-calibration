"""Single place for every path used by the scripts in this repository.

All data live under data/ and paper/data/ relative to the repository root, so a fresh clone
runs without edits. Two optional environment variables point at inputs that are not in the
repository:
  PNFAM_CALIB_RESULTS  output root for new MCMC runs (default: <repo>/results)
  PNFAM_RAW            root of the raw pnFAM outputs from the companion archive
                       pnfam-emulator-calibration-data (only extract_n126_ff.py and
                       table_n126_gs.py without its cache need it)
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

CALIBRATION = os.path.join(ROOT, "calibration")       # bayes_mcmc.py, config.py, run scripts
EMULATOR_CODE = os.path.join(ROOT, "emulator")        # model definition and training scripts
DESIGN = os.path.join(ROOT, "design")                 # set selection: sensitivities, tag tables, search outputs
QCORR = os.path.join(ROOT, "qcorr")                   # Q-value correction
QCORR_RESULTS = os.path.join(QCORR, "results")

DATA = os.path.join(ROOT, "data")
POSTERIORS = os.path.join(DATA, "posteriors")         # <F>/<set>_<likelihood>/{trace.nc,summary.csv,manifest.txt}
EMULATORS = os.path.join(DATA, "emulators")           # <F>/HL/<nuc>/params_best_*.txt, <F>/GT/<nuc>/full/params_best_*.txt
ERRORS = os.path.join(DATA, "errors")                 # test emulator errors, sigma_Q tables
CV = os.path.join(ERRORS, "cv")                       # <F>/HL/<nuc>/fold_k/parity_*.txt, <F>/GT/<nuc>/full/fold_k/...
TRAINING = os.path.join(DATA, "training")             # <F>/HL/<nuc>/{params.txt,halflives.txt}, <F>/GT/<nuc>/full/sample_*.dat
TEST_BATCH = os.path.join(DATA, "test_batch21001")    # <F>/HL/<nuc>/{params,halflives}_batch21001.txt
N126 = os.path.join(DATA, "n126")                     # <F>/<nuc>/batch{30001,30201}/ (N = 126 prediction runs)
EXPERIMENT = os.path.join(DATA, "experiment")         # NUBASE2020, AME2020

PAPER = os.path.join(ROOT, "paper")
PAPER_DATA = os.path.join(PAPER, "data")              # precomputed tables read by the figure scripts
LITERATURE = os.path.join(PAPER_DATA, "literature")   # published half-life tables used for comparison
FIGS = os.path.join(PAPER, "figs")
TABLES = os.path.join(PAPER, "tables")

RESULTS = os.environ.get("PNFAM_CALIB_RESULTS", os.path.join(ROOT, "results"))
PNFAM_RAW = os.environ.get("PNFAM_RAW", "")

FUNCTIONALS = ["SKOP", "SLY4", "HFB1"]               # SkO', SLy4, UNEDF1-HFB
LIKELIHOOD = {"t4gtt4": "studentt", "norm": "normal"}  # run-name tag of bayes_mcmc.py -> directory suffix

# Calibration sets of the paper and the keys they have in calibration/config.py.
# For UNEDF1-HFB the G10 and L10 sets are identical; only the G10 run exists.
SET_KEYS = {"SKOP": {"setA": "A", "G10": "G10_SKOP", "L10": "L10_SKOP", "full": "POOL40_SKOP"},
            "SLY4": {"setA": "A", "G10": "G10_SLY4", "L10": "L10_SLY4", "full": "POOL40_SLY4"},
            "HFB1": {"setA": "A", "G10": "G10_HFB1", "L10": "L10_HFB1", "full": "POOL40_HFB1"}}


def posterior(functional, set_name, likelihood):
    """Directory of a posterior used in the paper.

    set_name: 'setA', 'G10', 'L10' or 'full'; likelihood: 'studentt' or 'normal'
    (the bayes_mcmc.py tags 't4gtt4' and 'norm' are accepted too)."""
    likelihood = LIKELIHOOD.get(likelihood, likelihood)
    if functional == "HFB1" and set_name == "L10":
        set_name = "G10"
    return os.path.join(POSTERIORS, functional, f"{set_name}_{likelihood}")
