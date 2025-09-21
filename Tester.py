import os
import itertools
import cupy as np
import sys
from Maininjector import main

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Provide at least a sequence, e.g. python3 Tester.py QKASASAE")

    sequence = sys.argv[1]

    # valori opzionali con default
    gamma_values      = [float(sys.argv[2])] if len(sys.argv) > 2 else [0.05]
    eps_yukawa_values = [float(sys.argv[3])] if len(sys.argv) > 3 else [0.03, 0.8]
    eps_hb_values     = [float(sys.argv[4])] if len(sys.argv) > 4 else [0.5, 1.5]
    eq_iters          = int(sys.argv[5]) if len(sys.argv) > 5 else 500
    prod_iters        = int(sys.argv[6]) if len(sys.argv) > 6 else 1000

    # parametri fissi
    gridshape = (75, 75, 100, 60)
    max_iter = eq_iters+prod_iters
    rhop0_values = [0.16]
    vchi_ps_values = [1.0]
    vchi_pp_values = [0.0]
    bjerrum_values = [0.0]
    salt_fractions = [0.005]
    decay_yukawa = 2
    decay_es = 1

    param_combinations = list(itertools.product(
        rhop0_values, salt_fractions, vchi_ps_values, 
        vchi_pp_values, bjerrum_values, eps_yukawa_values, eps_hb_values
    ))
    print(f"\nTrying all parameter combinations for sequence: {sequence} ({len(param_combinations)} combinations)")

    for i, (rhop0, salt_fraction, vchi_ps, vchi_pp, bjerrum, eps_yukawa, epsilon_hb) in enumerate(param_combinations):
        for gamma in gamma_values:
            outdir = f"{sequence}/bj{bjerrum}_vps{vchi_ps}_epsy{eps_yukawa}_epshb{epsilon_hb}/g{gamma}"
            os.makedirs(outdir, exist_ok=True)
            print(f"\n[{i+1}/{len(param_combinations)}] {sequence}, rhop0={rhop0}, salt={salt_fraction}, "
                  f"gamma={gamma}, eps_yukawa={eps_yukawa}, epsilon_hb={epsilon_hb}")

            results = main(sequence, epsilon_hb, vchi_pp, vchi_ps, eps_yukawa,
                           decay_yukawa, bjerrum, decay_es, rhop0, max_iter,
                           gamma, salt_fraction, gridshape, eq_iters, prod_iters, outdir)

            PartialsHystory, DeltaRhoHystory, LDVCs, LDVCmaxs, LDVCmins, Broken, PS = results
