import cupy as np
class SCFTUpdater:
    def __init__(self, species_order, gridshape, vchi_pp, vchi_ps, epsilon_hb,bjerrum_length, hydro_charges, hydro_lambdas, es_charges,
                  n_history=15, reg=1e-12):
        self.species_order = species_order
        self.gridshape = gridshape
        self.vchi_ps = vchi_ps
        self.vchi_pp = vchi_pp
        self.epsilon_hb = epsilon_hb
        self.bjerrum_length = bjerrum_length
        self.hydro_charges = hydro_charges
        self.hydro_lambdas = hydro_lambdas
        self.es_charges = es_charges
        self.n_history = n_history
        self.reg = reg
        self.history_w = []
        self.history_d = []

    def linear_descent(self, xi_prior, w_prior_bb, w_prior_solv, w_prior_sc, w_prior_sc_rs, rho_bb_new, rho_solv_new, rho_sc_new, rho_sc_rs_new, gamma, ang_weights, spat_weights,  K_hydro, K_hb, A_hb, K_es, h_as, c_field, box_length, gridshape):
        dx = box_length[0]/gridshape[0]
        
        rhoPnew = np.zeros(gridshape[:3])
        for rhos in rho_bb_new:
            rhoPnew += np.sum(ang_weights*rho_bb_new[rhos], axis = -1)
        for rhos in rho_sc_new:
            rhoPnew += np.sum(ang_weights*rho_sc_new[rhos], axis = -1)
        for rhos in rho_sc_rs_new:
            rhoPnew += np.sum(ang_weights*rho_sc_rs_new[rhos], axis = -1)

        rhoS_total = sum(rho_solv_new[s] for s in ["neutral","plus","minus"])
        rho_tot = rhoPnew + rhoS_total
        
        ihatc = np.fft.ifftn(np.fft.fftn(c_field)*K_es).real * (dx**3)
        ihath_a = {a: np.fft.ifftn(np.fft.fftn(h_as[a])*K_hydro).real * (dx**3) for a in h_as} 
         
        xi =  xi_prior + 0.8*(rho_tot-1)
        wP_trial = {}
        for species in rho_bb_new:
            comp_vchi_ps, comp_vchi_pp = np.zeros(gridshape[:3]), np.zeros(gridshape[:3])
            if species != "pb":
                comp_vchi_ps = rhoS_total*self.vchi_ps
                comp_vchi_pp = rhoPnew*self.vchi_pp
            wP_trial[species] = np.broadcast_to(comp_vchi_ps[..., None] + comp_vchi_pp[..., None] , gridshape).copy()+np.broadcast_to(xi[..., None], gridshape).copy()
            wP_trial[species] -= w_prior_bb[species]

        wSc_trial = {} 
        for sc in rho_sc_new:
            contrib_rhoS = rhoS_total * self.vchi_ps
            contrib_hb = -0.5*self.epsilon_hb*np.fft.ifft2(np.fft.fft2(np.tensordot(rho_sc_new['Csc' if sc=='Nsc' else 'Nsc']*ang_weights[None,None,:], A_hb, axes=([3],[1])), axes=(0,1,2))*K_hb[..., None], axes=(0,1,2)).real*(dx**3)
            wSc_trial[sc] = (contrib_hb + np.broadcast_to(contrib_rhoS[..., None], gridshape).copy()+np.broadcast_to(xi[..., None], gridshape).copy())
            wSc_trial[sc] -= w_prior_sc[sc]
            
        WScRs_trial = {}
        for species in rho_sc_rs_new:
            comp_vchi_ps, comp_vchi_pp = np.zeros(gridshape[:3]), np.zeros(gridshape[:3])
            h_as_total = np.zeros_like(rhoPnew)
            for a_key in h_as:
                idx_in_res = next((i for i,res in enumerate(self.species_order) if res==species), None)
                if idx_in_res is not None:
                    h_as_total += self.hydro_lambdas[a_key]*self.hydro_charges[a_key][idx_in_res]*ihath_a[a_key]

            comp_vchi_ps = rhoS_total*self.vchi_ps
            comp_vchi_pp = rhoPnew*self.vchi_pp
            comp_h_as = h_as_total
            comp_es = self.bjerrum_length* self.es_charges[species]*ihatc
            WScRs_trial[species] = np.broadcast_to(comp_vchi_pp[..., None]+ comp_vchi_ps[..., None] + comp_h_as[..., None] + comp_es[..., None] +np.broadcast_to(xi[..., None], gridshape).copy(), gridshape).copy()
            WScRs_trial[species] -= w_prior_sc_rs[species]

        wS_trial = {}
        for s in rho_solv_new:
            contrib_rhoP = rhoPnew * self.vchi_ps
            contrib_es = self.es_charges[s] * ihatc
            wS_trial[s] = (contrib_rhoP + contrib_es+ xi)
            wS_trial[s] -= w_prior_solv[s]

        w_new_bb = {}
        for key in w_prior_bb:
            w_new_bb[key] = w_prior_bb[key] + gamma * (wP_trial[key])
        w_new_solv = {}
        for key in w_prior_solv:
            w_new_solv[key] = w_prior_solv[key] + gamma *( wS_trial[key] )
        w_new_sc = {}
        for key in w_prior_sc:
            w_new_sc[key] = w_prior_sc[key] + gamma*(wSc_trial[key] )
        w_new_sc_rs = {}
        for key in WScRs_trial:
            w_new_sc_rs[key] = w_prior_sc_rs[key] + gamma*(WScRs_trial[key])

        bb_deviation = sum(np.sum((wP_trial[key])**2 * spat_weights[..., None]*ang_weights[None, None, None, ...]) for key in wP_trial)
        sc_rs_deviation = sum(np.sum((WScRs_trial[key])**2 * spat_weights[..., None]*ang_weights[None, None, None,...]) for key in WScRs_trial)
        sc_deviation =  sum(np.sum((wSc_trial[key])**2 * spat_weights[..., None]*ang_weights[None, None, None,...]) for key in wSc_trial)
        solv_deviation = sum(np.sum((wS_trial[key])**2  * spat_weights) for key in w_new_solv)
        print(f'Deviations being: BB:= {bb_deviation}, SOLV:=  {solv_deviation}, SC:=  {sc_deviation}, SC_RS:={sc_rs_deviation}')
        return w_new_bb, w_new_sc, w_new_solv, w_new_sc_rs, xi, [bb_deviation, sc_deviation, solv_deviation, sc_rs_deviation]
        
    def zero_update(self, rho_bb_new, rho_solv_new, rho_sc_new, rho_sc_rs_class, gamma, ang_weights, spat_weights,  K_hydro, K_hb, A_hb, K_es, h_as, c_field, box_length, gridshape):
        rhoPnew = np.zeros(gridshape[:3])
        for key in rho_bb_new:
            rhoPnew += np.sum(rho_bb_new[key]*ang_weights, axis = -1)
        for key in rho_sc_new:
            rhoPnew += np.sum(rho_sc_new[key]*ang_weights, axis = -1)
        for key in rho_sc_rs_class:
            rhoPnew += np.sum(rho_sc_rs_class[key]*ang_weights, axis = -1)

        
        dx = box_length[0]/gridshape[0]
        rhoS_total = sum(rho_solv_new[s] for s in ["neutral","plus","minus"])
        
        ihatc = np.fft.ifftn(np.fft.fftn(c_field)*K_es).real * (dx**3)
        ihath_a = {a: np.fft.ifftn(np.fft.fftn(h_as[a])*K_hydro).real * (dx**3) for a in h_as} 
        rho_tot = rhoPnew + rhoS_total
        xi =  0.01*(1-1/rho_tot)

        wP_trial = {}
        for species in rho_bb_new:
            comp_vchi_ps, comp_vchi_pp = np.zeros(gridshape[:3]), np.zeros(gridshape[:3])
            if species != "pb":
                comp_vchi_ps = rhoS_total*self.vchi_ps
                comp_vchi_pp = rhoPnew*self.vchi_pp
            wP_trial[species] = gamma*np.broadcast_to(comp_vchi_ps[..., None] + comp_vchi_pp[..., None] +xi[..., None], gridshape).copy()

        wSc_trial = {} 
        for sc in rho_sc_new:
            contrib_rhoS = rhoS_total * self.vchi_ps
            contrib_hb = -0.5*self.epsilon_hb*np.fft.ifft2(np.fft.fft2(np.tensordot(rho_sc_new['Csc' if sc=='Nsc' else 'Nsc']*ang_weights[None,None,None,:], A_hb, axes=([3],[1])), axes=(0,1,2))*K_hb[..., None], axes=(0,1,2)).real*(dx**3)
            wSc_trial[sc] = gamma*(contrib_hb + np.broadcast_to(contrib_rhoS[..., None] + xi[..., None], gridshape).copy())

        WScRs_trial = {}
        for species in rho_sc_rs_class:
            comp_vchi_ps, comp_vchi_pp = np.zeros(gridshape[:3]), np.zeros(gridshape[:3])
            h_as_total = np.zeros_like(rhoPnew)
            for a_key in h_as:
                idx_in_res = next((i for i,res in enumerate(self.species_order) if res==species), None)
                if idx_in_res is not None:
                    h_as_total += self.hydro_lambdas[a_key]*self.hydro_charges[a_key][idx_in_res]*ihath_a[a_key]

            comp_vchi_ps = rhoS_total*self.vchi_ps
            comp_vchi_pp = rhoPnew*self.vchi_pp
            comp_h_as = h_as_total
            comp_es = self.bjerrum_length* self.es_charges[species]*ihatc
            WScRs_trial[species] = gamma*np.broadcast_to(comp_vchi_pp[..., None]+ comp_vchi_ps[..., None] + comp_h_as[..., None] + comp_es[..., None] +xi[..., None], gridshape).copy()

        wS_trial = {}
        for s in rho_solv_new:
            contrib_rhoP = rhoPnew * self.vchi_ps
            contrib_es = self.es_charges[s] * ihatc
            wS_trial[s] = gamma*(contrib_rhoP + contrib_es + xi)
        
        return wP_trial, wSc_trial, wS_trial, WScRs_trial, xi