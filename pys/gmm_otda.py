# gmm_otda.py
import torch
import numpy as np
from scipy.special import gamma, roots_hermite
from pys.prob_utils import diag_gmm_log_probs


class GMMOTDA:

    def __init__(self,
                 ot_solver,
                 clustering_source,
                 clustering_target,
                 min_var=0.01,
                 q_=2,
                 ):
        self.ot_solver = ot_solver
        self.clustering_source = clustering_source
        self.clustering_target = clustering_target
        self.min_var = min_var

        self.stds_src = None
        self.weights_src = None
        self.labels_src = None
        self.centroids_src = None

        self.stds_tgt = None
        self.labels_tgt = None
        self.weights_tgt = None
        self.centroids_tgt = None
        self.estimated_labels_tgt = None

        self.ot_plan = None
        self.n_dim = None
        self.n_classes = None

        self.fitted_gmm = False
        self.fitted_ot = False

        self.q_ = float(q_)
        self.A = np.sqrt(np.pi) / gamma((self.q_ + 1.0) / 2.0)

        self.is_q_1 = abs(self.q_ - 1.0) < 1e-9
        self.is_q_2 = abs(self.q_ - 2.0) < 1e-9

        self.c_p = None
        self.beta_p = None

        if not self.is_q_1 and not self.is_q_2:
            if 1.0 < self.q_ < 2.0:
                self.c_p = 4.0 * (gamma((self.q_ + 1.0) / 2.0) / np.sqrt(np.pi)) ** (4.0 / self.q_)
            elif 2.0 < self.q_ <= 3.0:
                self.c_p = 4.0 * (gamma((self.q_ + 1.0) / 2.0) / np.sqrt(np.pi)) ** (4.0 / self.q_)
                self.beta_p = self._compute_beta_p(self.q_)

    def __del__(self):
        self.weights_src = None
        self.centroids_src = None
        self.stds_src = None
        self.labels_src = None
        self.weights_tgt = None
        self.centroids_tgt = None
        self.stds_tgt = None
        self.labels_tgt = None
        self.estimated_labels_tgt = None
        self.ot_plan = None

    def _to_detached_tensor(self, x, device=None, dtype=None):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().to(device=device, dtype=dtype)
        else:
            return torch.from_numpy(np.asarray(x)).to(device=device, dtype=dtype)

    def _compute_beta_p(self, p):
        from scipy.special import gamma, hyp1f1
        from scipy.optimize import minimize_scalar

        cp_val = 4.0 * (gamma((p + 1.0) / 2.0) / np.sqrt(np.pi)) ** (4.0 / p)

        def neg_beta_func(log_x):
            x = np.exp(log_x)
            val_1f1 = hyp1f1(-p / 2.0, 0.5, -x)
            if val_1f1 <= 0:
                return 1e9
            numerator = cp_val * (val_1f1 ** (4.0 / p)) - 4.0 * (x ** 2) - cp_val
            return - (numerator / (2.0 * x))

        res = minimize_scalar(neg_beta_func, bounds=(-15, 15), method='bounded')
        max_beta = -res.fun

        limit_zero = 2.0 * cp_val
        limit_inf = 2.0 * (p - 1.0)
        return float(max(max_beta, limit_zero, limit_inf))

    def fit_gmms(self, Xs, Ys, Xt, Yt=None):
        self.n_dim = Xs.shape[1]
        self.n_classes = Ys.shape[1]
        device = Xs.device
        dtype = Xs.dtype

        w_s, m_s, v_s, y_s = self.clustering_source(Xs, Ys)
        
        self.weights_src = self._to_detached_tensor(w_s, device, dtype)
        self.centroids_src = self._to_detached_tensor(m_s, device, dtype)
        
        v_s_tensor = self._to_detached_tensor(v_s, device, dtype)
        v_s_tensor = torch.clamp(v_s_tensor, min=self.min_var)
        self.stds_src = v_s_tensor ** 0.5
        self.labels_src = self._to_detached_tensor(y_s, device, dtype)

        w_t, m_t, v_t, y_t = self.clustering_target(Xt, Yt)
        
        self.weights_tgt = self._to_detached_tensor(w_t, device, dtype)
        self.centroids_tgt = self._to_detached_tensor(m_t, device, dtype)
        
        v_t_tensor = self._to_detached_tensor(v_t, device, dtype)
        v_t_tensor = torch.clamp(v_t_tensor, min=self.min_var)
        self.stds_tgt = v_t_tensor ** 0.5
        self.labels_tgt = self._to_detached_tensor(y_t, device, dtype)

        self.fitted_gmm = True
        return self

    def fit_ot(self, num_gh_nodes=7, eps=1e-12):
        if not self.fitted_gmm:
            raise ValueError("Expected 'fit_gmm' to be previously called.")

        diff_m = self.centroids_src.unsqueeze(1) - self.centroids_tgt.unsqueeze(0)
        diff_s = self.stds_src.unsqueeze(1) - self.stds_tgt.unsqueeze(0)

        device = diff_m.device
        dtype = diff_m.dtype

        if self.is_q_1:
            abs_m = torch.abs(diff_m)
            abs_s = torch.abs(diff_s) + eps
            ratio = abs_m / (np.sqrt(2.0) * abs_s)
            expect_dim = (np.sqrt(2.0 / np.pi) * abs_s * torch.exp(-(ratio ** 2)) 
                          + abs_m * torch.erf(ratio))
            C = torch.sum(expect_dim, dim=-1)

        elif self.is_q_2:
            C = torch.sum(diff_m ** 2 + diff_s ** 2, dim=-1)

        elif 1.0 < self.q_ < 2.0:
            m2 = diff_m ** 2
            s2 = diff_s ** 2
            m4 = m2 ** 2
            s4 = s2 ** 2

            c_p_tensor = torch.tensor(self.c_p, dtype=dtype, device=device)
            term = c_p_tensor * s2 * (s2 + 2.0 * m2) + m4
            expect_dim = torch.pow(term + eps, self.q_ / 4.0)
            C = torch.sum(expect_dim, dim=-1)

        elif 2.0 < self.q_ <= 3.0:
            m2 = diff_m ** 2
            s2 = diff_s ** 2
            m4 = m2 ** 2
            s4 = s2 ** 2

            c_p_tensor = torch.tensor(self.c_p, dtype=dtype, device=device)
            beta_p_tensor = torch.tensor(self.beta_p, dtype=dtype, device=device)
            term = m4 + beta_p_tensor * m2 * s2 + c_p_tensor * s4
            expect_dim = torch.pow(term + eps, self.q_ / 4.0)
            C = torch.sum(expect_dim, dim=-1)

        else:
            nodes, weights = roots_hermite(num_gh_nodes)
            nodes = torch.tensor(nodes, dtype=dtype, device=device)
            weights = torch.tensor(weights, dtype=dtype, device=device)

            diff_m_exp = diff_m.unsqueeze(-1)
            diff_s_exp = diff_s.unsqueeze(-1)
            nodes_exp = nodes.view(1, 1, 1, -1)
            weights_exp = weights.view(1, 1, 1, -1)

            val = diff_m_exp + np.sqrt(2.0) * diff_s_exp * nodes_exp
            val_pow_p = torch.pow(val ** 2 + eps, self.q_ / 2.0)

            expect_per_dim = (1.0 / np.sqrt(np.pi)) * torch.sum(val_pow_p * weights_exp, dim=-1)
            C = torch.sum(expect_per_dim, dim=-1)

        self.ot_plan = self._to_detached_tensor(
            self.ot_solver(self.weights_src, self.weights_tgt, C),
            device=device,
            dtype=dtype
        )

        col_sum = self.ot_plan.sum(dim=0, keepdim=True)
        normalized_ot_plan = self.ot_plan / torch.clamp(col_sum, min=1e-20)
        self.estimated_labels_tgt = torch.mm(
            normalized_ot_plan.T,
            self.labels_src
        )
        self.fitted_ot = True
        return self

    def fit(self, Xs, Ys, Xt, Yt=None):
        return self.fit_gmms(Xs, Ys, Xt, Yt).fit_ot()

    def predict_target_components(self, X, return_proba=False):
        if not self.fitted_gmm:
            raise ValueError("Expected 'fit_gmm' to be called previously.")
        if not self.fitted_ot:
            raise ValueError("Expected 'fit_ot' to be called previously.")

        log_probs = diag_gmm_log_probs(
            X=X,
            weights=self.weights_tgt,
            means=self.centroids_tgt,
            stds=self.stds_tgt
        )
        log_proba_components = log_probs - log_probs.logsumexp(dim=0, keepdim=True)
        if return_proba:
            return log_proba_components.exp()
        return log_proba_components.argmax(dim=0)

    def predict_source_components(self, X, return_proba=False):
        if not self.fitted_gmm:
            raise ValueError("Expected 'fit_gmm' to be called previously.")

        log_probs = diag_gmm_log_probs(
            X=X,
            weights=self.weights_src,
            means=self.centroids_src,
            stds=self.stds_src
        )
        log_proba_components = log_probs - log_probs.logsumexp(dim=0, keepdim=True)
        if return_proba:
            return log_proba_components.exp()
        return log_proba_components.argmax(dim=0)

    def sample_from_source(self, n):
        if not self.fitted_gmm:
            raise ValueError("Expected 'fit_gmm' to be called previously.")

        device = self.centroids_src.device
        dtype = self.centroids_src.dtype

        probs = torch.clamp(self.weights_src, min=0.0)
        probs = probs / probs.sum()
        sampled_indices = torch.multinomial(probs, num_samples=n, replacement=True)

        means = self.centroids_src[sampled_indices]
        stds = self.stds_src[sampled_indices]

        Xsyn = stds * torch.randn(n, self.n_dim, device=device, dtype=dtype) + means

        class_indices = self.labels_src[sampled_indices].argmax(dim=-1)
        Ysyn = torch.nn.functional.one_hot(class_indices, num_classes=self.n_classes).to(device=device, dtype=dtype)

        return Xsyn, Ysyn

    def sample_from_target(self, n, use_estimated_labels=True):
        if not self.fitted_gmm:
            raise ValueError("Expected 'fit_gmm' to be called previously.")
        if use_estimated_labels and not self.fitted_ot:
            raise ValueError("Expected 'fit_ot' to be called previously.")
        if not use_estimated_labels and self.labels_tgt is None:
            raise ValueError(
                "If not using estimated labels, expects target GMM to be fitted using labels."
            )

        device = self.centroids_tgt.device
        dtype = self.centroids_tgt.dtype

        probs = torch.clamp(self.weights_tgt, min=0.0)
        probs = probs / probs.sum()
        sampled_indices = torch.multinomial(probs, num_samples=n, replacement=True)

        means = self.centroids_tgt[sampled_indices]
        stds = self.stds_tgt[sampled_indices]

        Xsyn = stds * torch.randn(n, self.n_dim, device=device, dtype=dtype) + means

        target_labels = self.estimated_labels_tgt if use_estimated_labels else self.labels_tgt
        class_indices = target_labels[sampled_indices].argmax(dim=-1)
        Ysyn = torch.nn.functional.one_hot(class_indices, num_classes=self.n_classes).to(device=device, dtype=dtype)

        return Xsyn, Ysyn

    def predict_source_labels(self, X):
        if not self.fitted_gmm:
            raise ValueError("Expected 'fit_gmm' to be called previously.")
        proba_components = self.predict_source_components(X, return_proba=True)
        cluster_labels = torch.mm(
            self.labels_src.T, proba_components).T
        return cluster_labels

    def predict_target_labels(self, X, use_estimated_labels=True):
        if not self.fitted_gmm:
            raise ValueError("Expected 'fit_gmm' to be called previously.")
        if not self.fitted_ot:
            raise ValueError("Expected 'fit_ot' to be called previously.")
        proba_components = self.predict_target_components(X, return_proba=True)
        if use_estimated_labels:
            cluster_labels = torch.mm(
                self.estimated_labels_tgt.T, proba_components).T
        else:
            cluster_labels = torch.mm(
                self.labels_tgt.T, proba_components).T
        return cluster_labels

    def compute_source_nll(self, X):
        if not self.fitted_gmm:
            raise ValueError("Expected 'fit_gmm' to be called previously.")

        log_probs = diag_gmm_log_probs(
            X=X,
            weights=self.weights_src,
            means=self.centroids_src,
            stds=self.stds_src
        )
        return - log_probs.logsumexp(dim=0).mean()

    def compute_target_nll(self, X):
        if not self.fitted_gmm:
            raise ValueError("Expected 'fit_gmm' to be called previously.")

        log_probs = diag_gmm_log_probs(
            X=X,
            weights=self.weights_tgt,
            means=self.centroids_tgt,
            stds=self.stds_tgt
        )
        return - log_probs.logsumexp(dim=0).mean()

    def transport_samples(self, X, Y, numel=None):
        device = X.device
        dtype = X.dtype

        if numel is not None and numel <= 0:
            return (torch.empty(0, dtype=dtype, device=device),
                    torch.empty((0, X.shape[1]), dtype=dtype, device=device),
                    torch.empty((0, Y.shape[1]), dtype=Y.dtype, device=Y.device))

        ot_plan_np = self.ot_plan.detach().cpu().numpy()
        if numel is None:
            numel = ot_plan_np.shape[0] + ot_plan_np.shape[1] - 1
        
        numel = min(numel, ot_plan_np.size)
        q = np.quantile(ot_plan_np.flatten(), 1 - numel / ot_plan_np.size)
        
        ind_s, ind_t = np.where((ot_plan_np >= q) & (ot_plan_np > 1e-20))

        transp_w = []
        transp_X = []
        transp_y = []

        components_s = self.predict_source_components(X)
        components_s_np = components_s.detach().cpu().numpy()

        for i_s, i_t in zip(ind_s, ind_t):
            idx = np.where(components_s_np == i_s)[0]
            if len(idx) == 0:
                continue

            x = X[idx]
            y = Y[idx]

            w = self.ot_plan[i_s, i_t]
            
            std_t = self.stds_tgt[i_t].to(device=device, dtype=dtype)
            std_s = self.stds_src[i_s].to(device=device, dtype=dtype)
            mean_t = self.centroids_tgt[i_t].to(device=device, dtype=dtype)
            mean_s = self.centroids_src[i_s].to(device=device, dtype=dtype)

            A = std_t / (std_s + 1e-9)
            b = mean_t - mean_s * A

            transp_w.append(torch.full((len(idx),), w.item(), dtype=dtype, device=device))
            transp_X.append(x * A + b)
            transp_y.append(y)

        if len(transp_X) == 0:
            return (torch.empty(0, dtype=dtype, device=device),
                    torch.empty((0, X.shape[1]), dtype=dtype, device=device),
                    torch.empty((0, Y.shape[1]), dtype=Y.dtype, device=Y.device))

        transp_w = torch.cat(transp_w, dim=0)
        transp_X = torch.cat(transp_X, dim=0)
        transp_y = torch.cat(transp_y, dim=0)

        return transp_w, transp_X, transp_y

    def rand_transport(self, X, Y, numel=None):
        device = X.device
        dtype = X.dtype

        proba_components_src = self.predict_source_components(X, return_proba=True).T

        ratio = self.ot_plan / (self.weights_src.view(-1, 1) + 1e-20)
        sampling_probs = ratio.unsqueeze(0) * proba_components_src.unsqueeze(2)

        sampling_probs_flat = sampling_probs.view(len(X), -1)

        p_sum = sampling_probs_flat.sum(dim=-1, keepdim=True)
        sampling_probs_flat = torch.where(
            p_sum > 1e-20,
            sampling_probs_flat / p_sum,
            torch.ones_like(sampling_probs_flat) / sampling_probs_flat.shape[-1]
        )

        sampled_flat_indices = torch.multinomial(sampling_probs_flat, num_samples=1).squeeze(-1)

        K_t = self.ot_plan.shape[1]
        k1 = torch.div(sampled_flat_indices, K_t, rounding_mode='floor')
        k2 = sampled_flat_indices % K_t

        stds_tgt = self.stds_tgt.to(device=device, dtype=dtype)
        stds_src = self.stds_src.to(device=device, dtype=dtype)
        centroids_tgt = self.centroids_tgt.to(device=device, dtype=dtype)
        centroids_src = self.centroids_src.to(device=device, dtype=dtype)

        A = stds_tgt[k2] / (stds_src[k1] + 1e-9)
        b = centroids_tgt[k2] - centroids_src[k1] * A

        mapped_x = X * A + b
        return mapped_x, Y