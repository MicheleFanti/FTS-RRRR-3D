import cupy as np
from Tools import *
from AAHydrophobic import *
from UpdateChemicalPotential import *
from PropagationStoch import *
import os
import time 


def main(sequence, epsilon_hb, vchi_pp, vchi_ps, eps_yukawa, decay_yukawa, bjerrum_length, decay_es, rhop0, max_iter, gamma, salt_fraction, gridshape, eq_iters, prod_iters, outdir):
    b_length = 0.38/3
    n_quad_per_rod = 6
    dx = 0.38/(3*n_quad_per_rod)
    
    l_hb = 0.18
    sigma_hb = 0.02 
    sigma_ev = 0.02
    gamma_min = gamma
    
    Nx, Ny, Nz, Nang = gridshape
    box_lengths = tuple(d * dx for d in gridshape[:3])
    V = np.prod(np.array(box_lengths))
    dV = V / (Nx * Ny* Nz)
    spat_weights = dV*np.ones((Nx, Ny, Nz))/V
    x = np.linspace(-box_lengths[0]/2,box_lengths[0]/2,Nx)
    y = np.linspace(-box_lengths[1]/2,box_lengths[1]/2,Ny)
    z = np.linspace(-box_lengths[1]/2,box_lengths[1]/2,Nz)
    X,Y, Z = np.meshgrid(x,y,z,indexing='ij')
    grid = np.array([X,Y, Z])

    l_chain = ((3* len(sequence)-1)/3)*0.38+2*0.15*len(sequence)
    for res in sequence:
        l_chain += 0.15*LateralChains.SideChain(res).length
        if LateralChains.SideChain(res).terminal == 'acceptor':
            l_chain +=0.15
        elif LateralChains.SideChain(res).terminal == 'donor':
            l_chain +=0.15
        elif LateralChains.SideChain(res).terminal == 'both':
            l_chain +=0.3
    u_vectors, ang_weights = lebedev_grid(Nang)
    u_vectors /= np.linalg.norm(u_vectors, axis=1)[:, None]
    ang_weights /= np.sum(ang_weights)
    
    residue_class_per_segment, res_classes, rho0_all, rhobb_class, rhosc_class, rho_sc_rs_class = build_initials(sequence, l_chain, rhop0, gridshape, spat_weights, ang_weights)
    charges, lambdas = gather_charges_and_lambdas(sequence)
    es_charges = get_es_charges(sequence)
    
    V_hydro, _ = make_3D_kernel_fft(yukawa_realspace, grid, spat_weights, eps_yukawa, decay_yukawa)
    V_es, _ = make_3D_kernel_fft(yukawa_realspace, grid, spat_weights, bjerrum_length, decay_es)
    V_hb, _    = make_3D_kernel_fft(gaussian_realspace, grid, spat_weights,  l_hb, sigma_hb)
    V_ev, _ = make_3D_kernel_fft(gaussian_realspace,  grid, spat_weights, 0, sigma_ev)
    
    A_hb = build_angular_kernel(Nang, u_vectors, np.pi, ang_weights)
    geom_kernel = build_angular_kernel(Nang, u_vectors, (2/3)*np.pi, ang_weights)
    antiparall_kernel = build_angular_kernel(Nang, u_vectors, np.pi, ang_weights)
    
    DeltaRhoHystory = []
    PartialsHystory = {k: [0] * (max_iter+1) for k in rho0_all}
    c_gamma = gamma
    LDVCs = []
    LDVCmins = []
    LDVCmaxs = []
    rhoS = {}
    rho0_sv = {}
    total_charge = sum(es_charges.get(res, 0.0) for res in sequence)
    save_interval = 10
    max_iter = eq_iters + prod_iters


    def computerhoS_zero(rho_class, rhop0, total_charge):
        rhoP_field = np.zeros(gridshape[:3])
        for key in rho0_all:
            rhoP_field += np.sum(rho_class[key]*ang_weights[None, None, None, ...], axis=-1)
        k = 1.0 - rhop0
        rho_s = salt_fraction * k
        counterion_fraction = np.abs(sum(es_charges[key] * Counter(sequence)[key] * rho0_all[key] for key in rho_class if es_charges[key] != 0))
        if counterion_fraction + rho_s > k:
            raise ValueError("Polymer charge + salt exceeds free fraction.")
            
        if total_charge >= 0:
            rho_minus = rho_s + counterion_fraction
            rho_plus = rho_s
        else:
            rho_plus = rho_s + counterion_fraction
            rho_minus = rho_s
        rho_neutral = k - rho_plus - rho_minus

        rho0_sv['plus'] = rho_plus
        rho0_sv['minus'] = rho_minus
        rho0_sv['neutral'] = rho_neutral

        scaling = 1.0 - rhoP_field
        rhoS['plus'] = rho_plus* scaling
        rhoS['minus'] = rho_minus * scaling
        rhoS['neutral'] = rho_neutral * scaling
        print(f'---> INITIAL DENSITIES -> plus: {rho_plus:.4f}, minus: {rho_minus:.4f}, neutral: {rho_neutral:.4f}, salt fraction: {rho_s:.4f}\n\n')
        return rhoS
    
    computerhoS_zero({**rhobb_class, **rhosc_class, **rho_sc_rs_class}, rhop0, total_charge)  
    Broken = False  
    PS = False
    c_gamma = gamma
    if outdir is None:
        outdir = f"density_outputs_{sequence}"
    os.makedirs(outdir, exist_ok=True)

    # subfolders
    logs_folder = os.path.join(outdir, "logs")
    plots_folder = os.path.join(outdir, "density_plots")
    npz_folder  = os.path.join(outdir, "saved_densities")
    propagator_folder  = os.path.join(outdir, "propagators")

    for folder in [logs_folder, plots_folder, npz_folder, propagator_folder]:
        os.makedirs(folder, exist_ok=True)

    # log file
    ldvc_log_filename = os.path.join(logs_folder, f"LDVCs_log_vchi_ps_{vchi_ps}_rhop0_{rhop0}_gamma_{gamma}.txt")
    with open(ldvc_log_filename, "w") as f:
        header = "iter\tLDVC_mean\tLDVC_max\tLDVC_min\t" + \
                "\t".join([f"Partial_{key}" for key in res_classes]) + \
                "\tDeviation\tQ\n"
        f.write(header)

    # initial plot before iterations
    plot_densities(sequence, {**rhobb_class, **rhosc_class}, rhoS, gridshape, -1,
                0.0, vchi_ps, gamma, 0.0, rhop0,
                eps_yukawa, bjerrum_length, ang_weights,
                plots_folder)
    ldvc_log_filename = os.path.join(logs_folder, f"LDVCs_log_vchi_pp_{0.0}_vchi_ps_{vchi_ps}_rhop0_{rhop0}_gamma_{gamma}.txt")
    with open(ldvc_log_filename, "w") as f:
        header = "iter\tLDVC_mean\tLDVC_max\tLDVC_min\t" + "\t".join([f"Partial_{key}" for key in res_classes]) + "\tDeviation_bb\tDeviation_solv\tDeviation_baresc\tDeviation_rs_sc\tQ\n"
        f.write(header)
    

    h_as = compute_has(sequence, charges, rhobb_class, res_classes, ang_weights, gridshape)
    c_field = compute_c(sequence, es_charges, rhobb_class, rhoS, res_classes, ang_weights, gridshape)
    
    mixer = SCFTUpdater(res_classes, gridshape, vchi_pp, vchi_ps, epsilon_hb, bjerrum_length, charges, lambdas, es_charges, n_history=5)
    w_chains, w_sidechains, w_solvent, w_sidechains_rs, xi = mixer.zero_update(rhobb_class, rhoS, rhosc_class, rho_sc_rs_class, gamma, ang_weights, spat_weights, V_hydro, V_hb, A_hb, V_es, h_as, c_field, box_lengths, gridshape)

    q_prev_fw_list = np.zeros((len(list(residue_class_per_segment)), n_quad_per_rod, *gridshape), dtype = np.complex64)
    q_prev_bw_list = np.zeros((len(list(residue_class_per_segment)), n_quad_per_rod, *gridshape), dtype = np.complex64)

    q_prev_fwsc_list = {key: np.zeros((n_quad_per_rod, *gridshape), dtype = np.complex64) for key in ['Nsc', 'Csc']}
    q_prev_bwsc_list = {key: np.zeros((n_quad_per_rod, *gridshape), dtype = np.complex64) for key in ['Nsc', 'Csc']}

    q_prev_fw_rs_sc_list = {key: np.zeros((LateralChains.SideChain(key).length, n_quad_per_rod, *gridshape), dtype=np.complex64) for key in set(sequence)}
    q_prev_bw_rs_sc_list = {key: np.zeros((LateralChains.SideChain(key).length, n_quad_per_rod, *gridshape), dtype=np.complex64) for key in set(sequence)}

    for it in range(1, max_iter+1):
        if it > 300:
            c_gamma = max(gamma_min, np.abs(gamma/((it-300))))
        start = time.time()
        h_as = compute_has(sequence, charges, rho_sc_rs_class, res_classes, ang_weights, gridshape)
        c_field = compute_c(sequence, es_charges, rho_sc_rs_class, rhoS, res_classes, ang_weights, gridshape)

        w_chains, w_sidechains, w_solvent, w_sidechains_rs, xi, deviations = mixer.linear_descent(xi, w_chains, w_solvent, w_sidechains, w_sidechains_rs, rhobb_class, rhoS, rhosc_class,rho_sc_rs_class, c_gamma,ang_weights,spat_weights, V_hydro, V_hb, A_hb, V_es, h_as, c_field, box_lengths, gridshape)
        if it == 1:
            rhobb_class, rhosc_class, rho_sc_rs_class, Q, q_prev_fw_list, q_prev_bw_list,q_prev_fwsc_list, q_prev_bwsc_list, q_prev_fw_rs_sc_list, q_prev_bw_rs_sc_list = propagate_closed(sequence, residue_class_per_segment, l_chain, rho0_all, w_chains, w_sidechains, w_sidechains_rs, ang_weights, spat_weights, u_vectors, gridshape, b_length, n_quad_per_rod, D_theta = 0.01**(-1), Lx=dx*Nx, Ly = dx*Ny, Lz = dx*Nz, dt=0.01, qf_previous= q_prev_fw_list, qb_previous= q_prev_bw_list, qf_prev_sc=q_prev_fwsc_list, qb_prev_sc=q_prev_bwsc_list,qf_prev_rs_sc=q_prev_fw_rs_sc_list, qb_prev_rs_sc=q_prev_bw_rs_sc_list, geom_kernel = geom_kernel, antiparallel_kernel=  antiparall_kernel, mode = 'deterministic')
        else:
            rhobb_class, rhosc_class, rho_sc_rs_class, Q, q_prev_fw_list, q_prev_bw_list,q_prev_fwsc_list, q_prev_bwsc_list, q_prev_fw_rs_sc_list, q_prev_bw_rs_sc_list = propagate_closed(sequence, residue_class_per_segment, l_chain, rho0_all, w_chains, w_sidechains, w_sidechains_rs, ang_weights, spat_weights, u_vectors, gridshape, b_length, n_quad_per_rod, D_theta = 0.01**(-1), Lx=dx*Nx, Ly = dx*Ny, Lz = dx*Nz, dt=0.01, qf_previous = q_prev_fw_list, qb_previous = q_prev_bw_list, qf_prev_sc=q_prev_fwsc_list, qb_prev_sc=q_prev_bwsc_list,qf_prev_rs_sc=q_prev_fw_rs_sc_list, qb_prev_rs_sc=q_prev_bw_rs_sc_list, geom_kernel = geom_kernel, antiparallel_kernel=  antiparall_kernel, mode = 'thermal')
        for solvents in ['plus', 'minus', 'neutral']:
            rhoS[solvents] =  rho0_sv[solvents]*np.exp(-w_solvent[solvents])/(np.sum(spat_weights*np.exp(-w_solvent[solvents])))

        rho_all_residue = {k: rhobb_class[k] + rho_sc_rs_class[k] if k in rho_sc_rs_class else rhobb_class[k] for k in rhobb_class}

        total_diff, partial_diffs = compute_constraint_violations(rho0_all, {**rho_all_residue, **rhosc_class}, spat_weights, ang_weights, V)
        DeltaRhoHystory.append(total_diff)
        
        local_density_violation_no_rhoSneutral = np.zeros(gridshape[:3], dtype = np.float64)
        for key in {**rhobb_class, **rhosc_class}:
            local_density_violation_no_rhoSneutral += np.sum({**rho_all_residue, **rhosc_class}[key] * ang_weights, axis=-1)
        LDVC_mean = np.mean(local_density_violation_no_rhoSneutral)
        LDVC_max = np.max(local_density_violation_no_rhoSneutral)
        LDVC_min = np.min(local_density_violation_no_rhoSneutral)

        partials_values = [partial_diffs[idx] for idx in range(len(res_classes))]
        deviation1, deviation2, deviation3, deviation4 = deviations  

        with open(ldvc_log_filename, "a") as f:
            f.write(f"{it}\t{LDVC_mean:.8f}\t{LDVC_max:.8f}\t{LDVC_min:.8f}\t")
            f.write("\t".join([f"{val:.8f}" for val in partials_values]))
            f.write(f"\t{deviation1:.8f}\t{deviation2:.8f}\t{deviation3:.8f}\t{deviation4:.8f}\t{Q:.8f}\n")
            
        if any(np.isnan({**rho_all_residue, **rhosc_class}[key]).any() for key in {**rhobb_class, **rhosc_class}) or any(np.isnan(rhoS[key]).any() for key in rhoS):
            Broken = True
            print(f'NaN detected at iteration {it}, simulation broken. GAMMA = {c_gamma:.4f},')
            break
        if it % 1 == 0:
            print(f'Iter {it} | Elapsed: {time.time()- start}|LDVC mean/max/min={LDVC_mean:.4f}/{LDVC_max:.4f}/{LDVC_min:.4f}, gamma = {c_gamma:.4f}, eps_yuk={eps_yukawa:.4f}, eps_hb = {epsilon_hb:.4f}, vchi_pp/ps = {vchi_pp}/{vchi_ps}, ')
        if it % 30 == 0:
            os.system('clear')   
            plot_densities(sequence, {**rho_all_residue, **rhosc_class}, rhoS, gridshape, it, vchi_pp, vchi_ps, gamma, 0.0, rhop0, eps_yukawa, bjerrum_length, ang_weights, plots_folder)
        if it > 1600 and it < 1700 and it %save_interval == 0:
            save_idx = it
            save_fname = os.path.join(npz_folder, f"densities_iter_{save_idx:04d}.npz")
            
            try:
                save_dict = {}

                # --- Save densities from rhoS, excluding solvent ---
                '''for s_key, s_arr in rhoS.items():
                    if s_key.lower() != "neutral":
                        save_dict[f"rhoS_{s_key}"] = (
                            s_arr.get().astype(np.float32) 
                            if np.isrealobj(s_arr) else s_arr.get()
                        )'''
                for a_key, a_arr in rhobb_class.items():  
                    if a_key != 'pb':
                        save_dict[f"{a_key}"] = (
                            a_arr.get().astype(np.float32) 
                            if np.isrealobj(a_arr) else a_arr.get()
                        )
                for a_key, a_arr in rhosc_class.items():  
                    save_dict[f"{a_key}"] = (
                        a_arr.get().astype(np.float32) 
                        if np.isrealobj(a_arr) else a_arr.get()
                    )
                for a_key, a_arr in rho_sc_rs_class.items():  
                    save_dict[f"{a_key}rs_sc"] = (
                        a_arr.get().astype(np.float32) 
                        if np.isrealobj(a_arr) else a_arr.get()
                    )
                np.savez_compressed(save_fname, **save_dict)
                print(f"Saved densities (excluding solvent) and l_p at iter {it} -> {save_fname}")

            except Exception as e:
                print(f"Warning: failed to save densities at iter {it}: {e}")
        elif it > 1800 and it < 1900 and it %save_interval == 0:
            save_idx = it
            save_fname = os.path.join(npz_folder, f"densities_iter_{save_idx:04d}.npz")
            
            try:
                save_dict = {}

                # --- Save densities from rhoS, excluding solvent ---
                '''for s_key, s_arr in rhoS.items():
                    if s_key.lower() != "neutral":
                        save_dict[f"rhoS_{s_key}"] = (
                            s_arr.get().astype(np.float32) 
                            if np.isrealobj(s_arr) else s_arr.get()
                        )'''
                for a_key, a_arr in rhobb_class.items():  
                    if a_key != 'pb':
                        save_dict[f"{a_key}"] = (
                            a_arr.get().astype(np.float32) 
                            if np.isrealobj(a_arr) else a_arr.get()
                        )
                for a_key, a_arr in rhosc_class.items():  
                    save_dict[f"{a_key}"] = (
                        a_arr.get().astype(np.float32) 
                        if np.isrealobj(a_arr) else a_arr.get()
                    )
                for a_key, a_arr in rho_sc_rs_class.items():  
                    save_dict[f"{a_key}rs_sc"] = (
                        a_arr.get().astype(np.float32) 
                        if np.isrealobj(a_arr) else a_arr.get()
                    )
                np.savez_compressed(save_fname, **save_dict)
                print(f"Saved densities (excluding solvent) and l_p at iter {it} -> {save_fname}")

            except Exception as e:
                print(f"Warning: failed to save densities at iter {it}: {e}")
        elif it > 2100 and it %15 == 0: 
            save_idx = it
            save_fname = os.path.join(npz_folder, f"densities_iter_{save_idx:04d}.npz")
            
            try:
                save_dict = {}

                # --- Save densities from rhoS, excluding solvent ---
                '''for s_key, s_arr in rhoS.items():
                    if s_key.lower() != "neutral":
                        save_dict[f"rhoS_{s_key}"] = (
                            s_arr.get().astype(np.float32) 
                            if np.isrealobj(s_arr) else s_arr.get()
                        )'''
                for a_key, a_arr in rhobb_class.items():  
                    if a_key != 'pb':
                        save_dict[f"{a_key}"] = (
                            a_arr.get().astype(np.float32) 
                            if np.isrealobj(a_arr) else a_arr.get()
                        )
                for a_key, a_arr in rhosc_class.items():  
                    save_dict[f"{a_key}"] = (
                        a_arr.get().astype(np.float32) 
                        if np.isrealobj(a_arr) else a_arr.get()
                    )
                for a_key, a_arr in rho_sc_rs_class.items():  
                    save_dict[f"{a_key}rs_sc"] = (
                        a_arr.get().astype(np.float32) 
                        if np.isrealobj(a_arr) else a_arr.get()
                    )
                np.savez_compressed(save_fname, **save_dict)
                print(f"Saved densities (excluding solvent) and l_p at iter {it} -> {save_fname}")

            except Exception as e:
                print(f"Warning: failed to save densities at iter {it}: {e}")

    return PartialsHystory, DeltaRhoHystory, LDVCs, LDVCmaxs, LDVCmins, Broken, PS
