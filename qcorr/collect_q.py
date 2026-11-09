#!/usr/bin/env python
"""
Collect HFB ground-state properties and Q_HFB from the survey outputs.

For every chain directory <SCRATCH>/<FUNC>/outputs/zZZZ/ pynfam writes
  meta/logfile_master.dat            one row per nucleus (chosen ground state)
  00000i/hfb_soln/thoout.dat         HFBTHO text output of the chosen ground state
  00000i/hfb_soln/hfb_meta/logfile.dat   one row per deformation point of the scan
The master log gives Energy, Def, HFB_Qval, EQRPA_max, E_gs (pynfam's own Q
estimate, Q = EQRPA_max - E_gs); thoout.dat gives lambda_n, lambda_p
("lambda (ala)") and the pairing gaps Delta_n, Delta_p ("delta(n,p)").

Output: results/Q_<FUNC>.txt with columns
  Z N A El Energy_MeV beta2 lam_n lam_p Delta_n Delta_p E_gs EQRPA_max Q_HFB_MeV HFB_Conv nconv/npts
and results/missing_<FUNC>.txt listing nuclei of nuclei_ee.txt with no result.

Usage: python collect_q.py [--set ee|oddA|oddodd] [FUNC ...]     (default ee; SKOP HFB1 SLY4)
Odd sets read <SCRATCH>/<set>/<F>/outputs and write results/<set>/Q_<F>.txt (kept apart from even-even).
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get('PNFAM_RAW_QSURVEY', '')   # raw HFB outputs of the Q survey (companion archive)
LISTS = {'ee': 'nuclei_ee.txt', 'oddA': 'nuclei_oddA.txt', 'oddodd': 'nuclei_oddodd.txt', 'shape_k2': 'nuclei_shape42.txt', 'shape_km2': 'nuclei_shape42.txt'}
LIST = os.path.join(HERE, 'nuclei_ee.txt')
RES = os.path.join(HERE, 'results')

RE_LAM = re.compile(r'lambda \(ala\) \.+\s+([-\d.]+)\s+([-\d.]+)')
RE_DEL = re.compile(r'delta\(n,p\), pwi \.+\s+([-\d.]+)\s+([-\d.]+)')


def read_list(path=LIST):
    out = {}
    with open(path) as fh:
        for ln in fh:
            t = ln.split()
            if ln.startswith('#') or t[0] == 'Z':
                continue
            out[(int(t[0]), int(t[1]))] = (int(t[2]), t[3])
    return out


def read_master(path):
    """Return list of dicts from a pynfam logfile_master.dat (whitespace table with index col)."""
    with open(path) as fh:
        lines = [ln.rstrip('\n') for ln in fh if ln.strip()]
    hdr = lines[0].split()
    rows = []
    for ln in lines[1:]:
        t = ln.split()
        if len(t) == len(hdr) + 1:      # pandas index column
            t = t[1:]
        if len(t) != len(hdr):
            continue
        rows.append(dict(zip(hdr, t)))
    return rows


def thoout_props(path):
    """lambda_n, lambda_p, Delta_n, Delta_p from the LAST occurrence in thoout.dat."""
    lam = dl = None
    with open(path, errors='replace') as fh:
        for ln in fh:
            m = RE_LAM.search(ln)
            if m:
                lam = (float(m.group(1)), float(m.group(2)))
            m = RE_DEL.search(ln)
            if m:
                dl = (float(m.group(1)), float(m.group(2)))
    return lam, dl


def scan_conv(path):
    """(n converged, n points) from hfb_meta/logfile.dat."""
    rows = read_master(path)
    return sum(1 for r in rows if r.get('HFB_Conv') == 'Yes'), len(rows)


def collect(func, nuclei, scratch=SCRATCH):
    res = {}
    for chain in sorted(glob.glob(os.path.join(scratch, func, 'outputs', 'z*'))):
        master = os.path.join(chain, 'meta', 'logfile_master.dat')
        if not os.path.isfile(master):
            continue
        for r in read_master(master):
            try:
                Z, N = int(r['Z']), int(r['N'])
            except (KeyError, ValueError):
                continue
            label = r['Label'].split('/')[0]
            soln = os.path.join(chain, label, 'hfb_soln')
            lam, dl = (None, None)
            if os.path.isfile(os.path.join(soln, 'thoout.dat')):
                lam, dl = thoout_props(os.path.join(soln, 'thoout.dat'))
            nconv, npts = (0, 0)
            if os.path.isfile(os.path.join(soln, 'hfb_meta', 'logfile.dat')):
                nconv, npts = scan_conv(os.path.join(soln, 'hfb_meta', 'logfile.dat'))

            def f(key):
                try:
                    return float(r[key])
                except (KeyError, ValueError):
                    return float('nan')
            res[(Z, N)] = dict(zblk=r.get('Z-Blk', '--'), nblk=r.get('N-Blk', '--'),
                               Energy=f('Energy'), Def=f('Def'), Q=f('HFB_Qval'),
                               Emax=f('EQRPA_max'), Egs=f('E_gs'), conv=r.get('HFB_Conv', '?'),
                               lam=lam or (float('nan'), float('nan')),
                               dl=dl or (float('nan'), float('nan')), nconv=nconv, npts=npts)
    return res


def main():
    args = sys.argv[1:]
    sset = 'ee'
    if args and args[0] == '--set':
        sset = args[1]; args = args[2:]
    funcs = args or ['SKOP', 'HFB1', 'SLY4']
    lst = os.path.join(HERE, LISTS[sset])
    res_dir = RES if sset == 'ee' else os.path.join(RES, sset)
    scratch = SCRATCH if sset == 'ee' else os.path.join(SCRATCH, sset)
    nuclei = read_list(lst)
    os.makedirs(res_dir, exist_ok=True)
    for func in funcs:
        res = collect(func, nuclei, scratch)
        out = os.path.join(res_dir, 'Q_%s.txt' % func)
        miss = os.path.join(res_dir, 'missing_%s.txt' % func)
        nmiss = 0
        with open(out, 'w') as fh, open(miss, 'w') as fm:
            fh.write('# HFB-only Q survey (%s), functional %s; Q_HFB = EQRPA_max - E_gs (pynfam, Engel 1999)\n' % (sset, func))
            fh.write('%4s %4s %4s %3s %12s %8s %8s %8s %7s %7s %8s %9s %9s %5s %5s %12s %12s\n' % (
                'Z', 'N', 'A', 'El', 'Energy_MeV', 'beta2', 'lam_n', 'lam_p', 'Del_n', 'Del_p',
                'E_gs', 'EQRPA_max', 'Q_HFB_MeV', 'Conv', 'nc/np', 'Z-Blk', 'N-Blk'))
            for (Z, N), (A, el) in sorted(nuclei.items()):
                if (Z, N) not in res:
                    fm.write('%4d %4d %4d %3s\n' % (Z, N, A, el))
                    nmiss += 1
                    continue
                r = res[(Z, N)]
                fh.write('%4d %4d %4d %3s %12.4f %8.4f %8.4f %8.4f %7.3f %7.3f %8.4f %9.4f %9.4f %5s %2d/%-2d %12s %12s\n' % (
                    Z, N, A, el, r['Energy'], r['Def'], r['lam'][0], r['lam'][1], r['dl'][0], r['dl'][1],
                    r['Egs'], r['Emax'], r['Q'], r['conv'], r['nconv'], r['npts'], r['zblk'], r['nblk']))
        extra = [zn for zn in res if zn not in nuclei]
        print('%s %s: %d of %d nuclei collected, %d missing (%s)%s' % (
            sset, func, len(nuclei) - nmiss, len(nuclei), nmiss, miss,
            ', %d extra not in list' % len(extra) if extra else ''))


if __name__ == '__main__':
    main()
