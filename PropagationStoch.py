import cupy as np
from cupy.fft import fftn, ifftn, fftfreq
from collections import Counter
import LateralChains
from scipy.special import sph_harm

class LebedevSHT:
    def __init__(self, points, weights, Lmax):
        x, y, z = points.T
        theta = np.arccos(z)
        phi = np.arctan2(y, x)
        self.w = np.asarray(weights)
        self.Lmax = Lmax
        Ylist = []
        for l in range(Lmax+1):
            for m in range(-l, l+1):
                Ylm = sph_harm(m, l, phi, theta)
                Ylist.append(Ylm)
        Y = np.stack(Ylist, axis=0)
        self.Y = np.asarray(Y)
        self.Y_conj = np.conj(self.Y)

    @property
    def nlm(self):
        return (self.Lmax+1)**2

    def forward(self, f):
        fw = f * self.w
        coeffs = np.tensordot(self.Y_conj, fw, axes=(1,3))
        return np.moveaxis(coeffs, 0, -1)

    def inverse(self, coeffs):
        coeffs = np.moveaxis(coeffs, -1, 0)
        f = np.tensordot(coeffs, self.Y, axes=(0,0))
        return f
def strang_step_wlc(q, w, ds, KX, KY, KZ, UX, UY, UZ, ang_mul_half, sht):
    N_ang = q.shape[-1]
    ang_mul_half_full = np.concatenate([np.repeat(ang_mul_half[l], 2*l+1) for l in range(len(ang_mul_half))])

    phase_half = np.exp(-1j * (
        KX[..., None]*UX[None, None, None, :] +
        KY[..., None]*UY[None, None, None, :] +
        KZ[..., None]*UZ[None, None, None, :]
    ) * (ds/2))
    q = ifftn(fftn(q, axes=(0,1,2)) * phase_half, axes=(0,1,2))
    flm = sht.forward(q* np.exp(-w*ds/2))
    flm = flm * ang_mul_half_full[None, :]
    q = sht.inverse(flm)
    q = q* np.exp(-w*ds/2)
    q = ifftn(fftn(q, axes=(0,1,2)) * phase_half, axes=(0,1,2))
    return q


def strang_step_wlc_backward(q, w, ds, KX, KY, KZ, UX, UY, UZ, ang_mul_half, sht):
    N_ang = q.shape[-1]
    ang_mul_half_full = np.concatenate([np.repeat(ang_mul_half[l], 2*l+1) for l in range(len(ang_mul_half))])

    phase_half = np.exp(1j * (
        KX[..., None]*UX[None, None, None, :] +
        KY[..., None]*UY[None, None, None, :] +
        KZ[..., None]*UZ[None, None, None, :]
    ) * (ds/2))

    q = ifftn(fftn(q, axes=(0,1,2)) * phase_half, axes=(0,1,2))
    flm = sht.forward(q* np.exp(-w*ds/2))
    flm = flm * ang_mul_half_full[None, :]
    q = sht.inverse(flm)
    q = q* np.exp(-w*ds/2)
    q = ifftn(fftn(q, axes=(0,1,2)) * phase_half, axes=(0,1,2))
    return q

def propagate_forward_wlc(q0_spatial, w, U_vectors, length, n_substeps, Dtheta, Lx, Ly, Lz, mu_forward, dt, q_prev, mode, sht):
    Nx, Ny, Nz = q0_spatial.shape[:3]
    N_ang = U_vectors.shape[0]
    ds = length / n_substeps
    kx = 2*np.pi*fftfreq(Nx, d=Lx/Nx)
    ky = 2*np.pi*fftfreq(Ny, d=Ly/Ny)
    kz = 2*np.pi*fftfreq(Nz, d=Lz/Nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    UX = np.asarray(U_vectors[:,0])
    UY = np.asarray(U_vectors[:,1])
    UZ = np.asarray(U_vectors[:,2])
    L = int(np.sqrt(N_ang) - 1)
    ell = np.arange(L+1)
    ang_mul_half = np.exp(-Dtheta*ell*(ell+1)*ds)
    if q0_spatial.ndim == 3 and q0_spatial.shape[-1] == 1:
        q_curr = np.repeat(q0_spatial, N_ang, axis=-1).astype(np.complex64)
    else:
        q_curr = np.asarray(q0_spatial, dtype=np.complex64)
    q_full = np.zeros((n_substeps, Nx, Ny, Nz, N_ang), dtype=np.complex64)
    w_arr = w if w.ndim==3 else np.repeat(w[:,:,:,None], N_ang, axis=-1)
    for i in range(n_substeps):
        if mode == 'thermal':
            q_curr = strang_step_wlc(q_curr - dt*(q_curr - q_prev[i]) + dt*mu_forward[...,i], w_arr, ds, KX, KY, KZ, UX, UY, UZ, ang_mul_half, sht)
        else:
            q_curr = strang_step_wlc(q_curr, w_arr, ds, KX, KY, KZ, UX, UY, UZ, ang_mul_half,sht)
        q_full[i] = q_curr
    return np.real(q_full)

def propagate_backward_wlc(q0_spatial, w, U_vectors, length, n_substeps, Dtheta, Lx, Ly, Lz, mu_backward, dt, q_prev, mode, sht):
    Nx, Ny, Nz = q0_spatial.shape[:3]
    N_ang = U_vectors.shape[0]
    ds = length / n_substeps
    kx = 2*np.pi*fftfreq(Nx, d=Lx/Nx)
    ky = 2*np.pi*fftfreq(Ny, d=Ly/Ny)
    kz = 2*np.pi*fftfreq(Nz, d=Lz/Nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    UX = np.asarray(U_vectors[:,0])
    UY = np.asarray(U_vectors[:,1])
    UZ = np.asarray(U_vectors[:,2])
    L = int(np.sqrt(N_ang) - 1)
    ell = np.arange(L+1)
    ang_mul_half = np.exp(-Dtheta*ell*(ell+1)*ds)
    if q0_spatial.ndim == 3 and q0_spatial.shape[-1] == 1:
        q_curr = np.repeat(q0_spatial, N_ang, axis=-1).astype(np.complex64)
    else:
        q_curr = np.asarray(q0_spatial, dtype=np.complex64)
    q_full = np.zeros((n_substeps, Nx, Ny, Nz, N_ang), dtype=np.complex64)
    w_arr = w if w.ndim==3 else np.repeat(w[:,:,:,None], N_ang, axis=-1)
    for i in range(n_substeps):
        if mode == 'thermal':
            q_curr = strang_step_wlc_backward(q_curr - dt*(q_curr - q_prev[i]) + np.sqrt(dt)*mu_backward[...,i], w_arr, ds, KX, KY, KZ, UX, UY, UZ, ang_mul_half, sht)
        else:
            q_curr = strang_step_wlc_backward(q_curr, w_arr, ds, KX, KY, KZ, UX, UY, UZ, ang_mul_half, sht)
        q_full[i] = q_curr
    return np.real(q_full)


def propagate_closed(sequence, backbone_seq, l_chain, rho0_per_class, w_per_class, w_sc, w_rs_sc, ang_weights, spat_weights, u_grid, gridshape, length_rod, n_quad_per_rod, D_theta, Lx, Ly, Lz, dt, qf_previous, qb_previous, qf_prev_sc, qb_prev_sc, qf_prev_rs_sc, qb_prev_rs_sc, geom_kernel, antiparallel_kernel, mode):
    sht = LebedevSHT(u_grid, ang_weights, 11)

    seq = list(backbone_seq)
    Nx, Ny, Nz, Nang = gridshape
    N = len(seq)
    sequence_length = int((N+1)/3)

    sc_keys = tuple(set(sequence)) #unique elements of the seqquence

    qf_list = np.zeros((N, n_quad_per_rod, *gridshape), dtype = np.complex64)
    q_init_spatial = np.ones(gridshape[:3], dtype = np.complex64)
    eta_1_full = np.random.randn(Nx, Ny, Nz, N, n_quad_per_rod) * np.sqrt(2)
    eta_2_full = np.random.randn(Nx, Ny, Nz, N, n_quad_per_rod) * np.sqrt(2)
    eta_sc1_full = {key: np.random.randn(Nx, Ny, Nz, n_quad_per_rod) * np.sqrt(2) for key in ['Nsc', 'Csc']}
    eta_sc2_full = {key: np.random.randn(Nx, Ny, Nz, n_quad_per_rod) * np.sqrt(2) for key in ['Nsc', 'Csc']}
    eta_rs_sc1_full = {key: np.random.randn(Nx, Ny, Nz, LateralChains.SideChain(key).length, n_quad_per_rod) * np.sqrt(2) for key in sc_keys}
    eta_rs_sc2_full = {key: np.random.randn(Nx, Ny, Nz, LateralChains.SideChain(key).length, n_quad_per_rod) * np.sqrt(2) for key in sc_keys}

    #Propagate elementar sidechains:
    q_sc_forward = {key: np.zeros((n_quad_per_rod, *gridshape), dtype=np.complex64) for key in ['Nsc', 'Csc']}
    for idx in ['Nsc', 'Csc']:
        eta_1 = eta_sc1_full[idx]
        eta_2 = eta_sc2_full[idx]
        mu_forward = (eta_1 + 1j*eta_2)/np.sqrt(2)
        mu_forward = np.broadcast_to(mu_forward[:, :, :, None, :], (Nx, Ny, Nz, Nang, n_quad_per_rod))
        q_sc_forward[idx] = propagate_forward_wlc(np.ones(gridshape[:3]), w_sc[idx], u_grid, length_rod, n_quad_per_rod, D_theta, Lx, Ly, Lz, mu_forward, dt, qf_prev_sc[idx], mode, sht)

    #Propagate residue-specific sidechains:
    q_sc_rs_forward = {key: np.zeros((LateralChains.SideChain(key).length, n_quad_per_rod, *gridshape), dtype=np.complex64) for key in sc_keys}
    for key in sc_keys:
        if LateralChains.SideChain(key).length != 0:
            for rod_element in range(LateralChains.SideChain(key).length):
                eta_1 = eta_rs_sc1_full[key][:, :, :, rod_element, :]
                eta_2 = eta_rs_sc2_full[key][:, :, :, rod_element, :]
                mu_forward = (eta_1 + 1j*eta_2)/np.sqrt(2)
                mu_forward = np.broadcast_to(mu_forward[:, :, :, None, :], (Nx, Ny, Nz,  Nang, n_quad_per_rod))

                if LateralChains.SideChain(key).terminal == 'donor':
                    q_sc_rs_forward[key][rod_element] = propagate_forward_wlc(np.sum(q_sc_forward['Nsc'][-1]*ang_weights, axis = -1), w_rs_sc[key], u_grid, length_rod, n_quad_per_rod, D_theta, Lx, Ly, Lz, mu_forward, dt, qf_prev_rs_sc[key][rod_element], mode, sht)
                elif LateralChains.SideChain(key).terminal == 'acceptor':
                    q_sc_rs_forward[key][rod_element] = propagate_forward_wlc(np.sum(q_sc_forward['Csc'][-1]*ang_weights, axis = -1), w_rs_sc[key], u_grid, length_rod, n_quad_per_rod, D_theta, Lx, Ly, Lz, mu_forward, dt, qf_prev_rs_sc[key][rod_element], mode, sht)
                elif LateralChains.SideChain(key).terminal == 'both':
                    ID = np.tensordot(q_sc_forward['Nsc'][-1] * ang_weights[None,None,None,:], geom_kernel, axes=([3],[0]))*np.tensordot(q_sc_forward['Csc'][-1] * ang_weights[None,None,None,:], geom_kernel, axes=([3],[0]))
                    q_sc_rs_forward[key][rod_element] = propagate_forward_wlc(ID, w_rs_sc[key], u_grid, length_rod, n_quad_per_rod, D_theta, Lx, Ly,  Lz, mu_forward, dt, qf_prev_rs_sc[key][rod_element], mode, sht)
                elif LateralChains.SideChain(key).terminal == 'none':
                    q_sc_rs_forward[key][rod_element] = propagate_forward_wlc(np.ones(gridshape[:3]), w_rs_sc[key], u_grid, length_rod, n_quad_per_rod, D_theta, Lx, Ly, Lz, mu_forward, dt, qf_prev_rs_sc[key][rod_element], mode, sht)
            

    #Propagate main backbone forward
    for idx in range(N):
        eta_1 = eta_1_full[:, :, :, idx, :]   
        eta_2 = eta_2_full[:, :, :, idx, :]
        mu_forward = (eta_1 + 1j*eta_2)/np.sqrt(2)
        mu_forward = np.broadcast_to(mu_forward[:, :, :, None, :], (Nx, Ny, Nz, Nang, n_quad_per_rod))
        res = seq[idx]
        if idx == 0: 
            q_init_spatial = np.tensordot(q_sc_forward['Nsc'][-1] * ang_weights[None,None,None, :], geom_kernel, axes=([3],[0]))
        elif idx % 3 == 0:
            q_init_spatial *= np.tensordot(q_sc_forward['Nsc'][-1] * ang_weights[None,None,None,:], geom_kernel, axes=([3],[0]))
        elif idx %3 == 1 and seq[idx] != 'G':
            q_init_spatial *= np.tensordot(q_sc_rs_forward[seq[idx]][-1, -1]* ang_weights[None,None,None,:], geom_kernel, axes=([3],[0]))
        elif idx % 3 == 2: 
            q_init_spatial *= np.tensordot(q_sc_forward['Csc'][-1] * ang_weights[None,None,None,:], geom_kernel, axes=([3],[0]))
        qf_list[idx] = propagate_forward_wlc(q_init_spatial, w_per_class[res], u_grid, length_rod, n_quad_per_rod, D_theta, Lx, Ly, Lz, mu_forward, dt, qf_previous[idx], mode, sht)
        q_init_spatial = np.tensordot(qf_list[idx][-1] * ang_weights[None,None,None,:], geom_kernel, axes=([3],[0]))

    #Backward propagation backbone, residual specific accumulation, partial bare sidechains accumulation
    qb_list = np.zeros((N, n_quad_per_rod, *gridshape), dtype = np.complex64)
    
    q_prev_spatial = np.ones(gridshape, dtype = np.complex64)
    q_sc_bb = {key: np.zeros(gridshape, dtype = np.complex64) for key in ['Nsc', 'Csc']}
    q_sc_rs_backward = {key: np.zeros((LateralChains.SideChain(key).length, n_quad_per_rod, *gridshape), dtype=np.complex64) for key in sc_keys}

    prev = np.zeros(gridshape, dtype = np.complex64) 
    curr = np.zeros(gridshape, dtype = np.complex64) 

    for idx in range(N-1, -1, -1):
        eta_1 = eta_1_full[:, :, :,idx, :]
        eta_2 = eta_2_full[:, :, :,idx, :]
        eta_1 = eta_1[:, :, :,::-1]
        eta_2 = eta_2[:, :, :,::-1]
        mu_backward = (eta_2 + 1j*eta_1)/np.sqrt(2)
        mu_backward = np.broadcast_to(mu_backward[:, :, :, None, :], (Nx, Ny, Nz, Nang, n_quad_per_rod))
        
        res = seq[idx]
        if idx == N-1:
            q_prev_spatial = np.tensordot(q_sc_forward['Csc'][-1] * ang_weights[None,None,None,:], geom_kernel, axes=([3],[0]))
        elif idx % 3 == 0:
            q_prev_spatial *= np.tensordot(q_sc_forward['Nsc'][-1] * ang_weights[None,None,None,:], geom_kernel, axes=([3],[0]))
        elif idx % 3 == 2:
            q_prev_spatial *= np.tensordot(q_sc_forward['Csc'][-1] * ang_weights[None,None,None,:], geom_kernel, axes=([3],[0]))
        elif idx %3 == 1 and seq[idx] != 'G':
            q_prev_spatial *= np.tensordot(q_sc_rs_forward[seq[idx]][-1, -1]* ang_weights[None,None,None,:], geom_kernel, axes=([3],[0]))

        qb_list[idx] = propagate_backward_wlc(q_prev_spatial, w_per_class[res], u_grid, length_rod, n_quad_per_rod, D_theta, Lx, Ly,  Lz, mu_backward, dt, qb_previous[idx], mode, sht)
        q_prev_spatial = np.tensordot(qb_list[idx][-1] * ang_weights[None,None,None,:], geom_kernel, axes=([3],[0]))
        #breakpoint()
        if idx == 0:
            q_sc_bb['Nsc'] += q_prev_spatial
        elif idx == N-1:
            q_sc_bb['Csc'] += np.tensordot(qf_list[N-1][-1] * ang_weights[None,None,None,:], geom_kernel, axes=([3],[0]))
        if idx % 3 == 2:
            curr = np.tensordot(((prev/(np.sum(prev*ang_weights, axis = -1)[..., None])))*ang_weights[None, None, None,:], antiparallel_kernel, axes = ([3], [0]))
            q_sc_bb['Csc'] += q_prev_spatial * np.tensordot(qf_list[idx-1][-1] * ang_weights[None,None,None,:], geom_kernel, axes=([3],[0]))*curr
        elif idx % 3 == 0:
            prev = q_prev_spatial * np.tensordot(qf_list[idx-1][-1] * ang_weights[None,None,None,:], geom_kernel, axes=([3],[0]))
            q_sc_bb['Nsc'] += prev
        elif idx % 3 == 1 and seq[idx] != 'G':
            q_sc_rs_backward[seq[idx]][-1, -1] += np.tensordot(qf_list[idx-1][-1] * ang_weights[None,None,None,:], geom_kernel, axes=([3],[0]))*q_prev_spatial
        
    #Backpropagation of residue specific sidechains
    for key in q_sc_rs_backward:
        if LateralChains.SideChain(key).length != 0:
            for rod_element in range(LateralChains.SideChain(key).length-1, -1, -1):
                eta_1 = eta_rs_sc1_full[key][:, :, :, rod_element, ::-1]
                eta_2 = eta_rs_sc2_full[key][:, :, :, rod_element, ::-1]
                mu_backward = (eta_2 + 1j*eta_1)/np.sqrt(2)
                mu_backward = np.broadcast_to(mu_backward[:, :, :, None, :], (Nx, Ny, Nz,Nang, n_quad_per_rod))
                if rod_element == LateralChains.SideChain(key).length-1:
                    q_sc_rs_backward[key][rod_element, :, :, :] = propagate_backward_wlc(q_sc_rs_backward[key][rod_element, -1], w_rs_sc[key], u_grid, length_rod, n_quad_per_rod, D_theta, Lx, Ly,  Lz, mu_backward, dt, qb_prev_rs_sc[key][rod_element], mode, sht)
                elif rod_element < LateralChains.SideChain(key).length-1:
                    q_sc_rs_backward[key][rod_element, :, :, :] = propagate_backward_wlc(q_sc_rs_backward[key][rod_element+1, 0], w_rs_sc[key], u_grid, length_rod, n_quad_per_rod, D_theta, Lx, Ly,  Lz, mu_backward, dt, qb_prev_rs_sc[key][rod_element], mode, sht)
            if LateralChains.SideChain(key).terminal == 'acceptor':
                q_sc_bb['Csc'] += np.sum(ang_weights*q_sc_rs_backward[key][0, 0], axis = -1)[..., None]
            elif LateralChains.SideChain(key).terminal == 'donor':
                q_sc_bb['Nsc'] += np.sum(ang_weights*q_sc_rs_backward[key][0, 0], axis = -1)[..., None]
            elif LateralChains.SideChain(key).terminal == 'both':
                ID = np.tensordot(ang_weights[None, None, None, ...]*q_sc_rs_backward[key][0, 0], geom_kernel, axes = ([3], [0]))
                q_sc_bb['Nsc'] += ID
                q_sc_bb['Csc'] += ID
        
    q_sc_bb_full = {key: np.zeros((n_quad_per_rod, *gridshape), dtype = np.complex64) for key in ['Nsc', 'Csc']}
    for idx in ['Nsc', 'Csc']:
        eta_1 = eta_sc1_full[idx][:, : , :, ::-1]
        eta_2 = eta_sc2_full[idx][:, : , :, ::-1]
        mu_backward = (eta_2 + 1j*eta_1)/np.sqrt(2)
        mu_backward = np.broadcast_to(mu_backward[:, :,:, None, :], (Nx, Ny, Nz, Nang, n_quad_per_rod))
        q_sc_bb_full[idx] = propagate_backward_wlc(q_sc_bb[idx], w_sc[idx], u_grid, length_rod, n_quad_per_rod, D_theta, Lx, Ly,  Lz, mu_backward, dt, qb_prev_sc[idx], mode, sht)
    
    temp_f = np.sum(qf_list[-1][-1] * ang_weights[None,None,None,:], axis=-1)
    
    Q = np.real(np.sum(temp_f * spat_weights))
    Q_check = np.real(np.sum(np.sum(qb_list[0][-1] * ang_weights[None,None,None,:], axis=-1) * spat_weights))
    
    print(f"Q = {Q}, Q_check = {Q_check}; rel.err = {(Q - Q_check)/Q_check}; additional check: {np.imag(Q)/np.real(Q)}, {np.imag(Q_check)/np.real(Q_check)}")
    
    rho_bb = {res: np.zeros(gridshape, dtype=np.float64) for res in set(seq)}
    rho_sc = {res: np.zeros(gridshape, dtype=np.float64) for res in ['Nsc', 'Csc']}
    rho_sc_rs = {res: np.zeros(gridshape, dtype=np.float64) for res in sc_keys}
    
    bb_segm_eff_length = (5/12)*0.38
    rod_length = 0.15
    
    for idx in range(N):
        res = seq[idx]
        qf = qf_list[idx]
        qb = qb_list[idx]
        for s in range(n_quad_per_rod):
            if seq[idx] != 'pb':
                rho_bb[res] += np.real((rho0_per_class[res]/(n_quad_per_rod*Counter(seq)[res]*Q_check))*np.real(qf[s] * qb[s]))*((2*bb_segm_eff_length)/((2*bb_segm_eff_length)+(LateralChains.SideChain(seq[idx]).length*0.15)))
            else: 
                rho_bb[res] += np.real((rho0_per_class[res]/(n_quad_per_rod*Counter(seq)[res]*Q_check))*np.real(qf[s] * qb[s]))
    n_app = {key: sequence_length for key in ['Nsc', 'Csc']}
    
    for key in sequence:
        if LateralChains.SideChain(key).terminal == 'acceptor':
            n_app['Csc'] += 1
        if LateralChains.SideChain(key).terminal == 'donor':
            n_app['Nsc'] += 1
        if LateralChains.SideChain(key).terminal == 'both':
            n_app['Csc'] += 1
            n_app['Nsc'] += 1
    for idx in rho_sc:
        for s in range(n_quad_per_rod):
            rho_sc[idx] += (rho0_per_class[idx]/(n_quad_per_rod*(n_app[idx])*(Q_check)))*np.real(q_sc_forward[idx][s] * q_sc_bb_full[idx][s])
        #print(f'sc {idx} {np.sum(np.sum(ang_weights*rho_sc[idx], axis = -1)*spat_weights)}')
    for key in rho_sc_rs:
        if key != 'G':
            for m in range(LateralChains.SideChain(key).length):
                for s in range(n_quad_per_rod):
                    rho_sc_rs[key] += (rho0_per_class[key]/(n_quad_per_rod*LateralChains.SideChain(key).length*Counter(seq)[key]*0.5*(Q_check)))*np.real(q_sc_rs_forward[key][m, s] * q_sc_rs_backward[key][m, s]) *((LateralChains.SideChain(key).length*0.15)/((2*bb_segm_eff_length)+(LateralChains.SideChain(key).length*0.15)))
        #print(f'rs_sc {key} {np.sum(np.sum(ang_weights*rho_sc_rs[key], axis = -1)*spat_weights)}')
    return rho_bb, rho_sc, rho_sc_rs, Q, qf_list, qb_list, q_sc_forward, q_sc_bb_full, q_sc_rs_forward, q_sc_rs_backward


