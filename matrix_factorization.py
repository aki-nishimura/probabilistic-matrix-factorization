import numpy as np
import scipy as scipy
import scipy.linalg
import scipy.sparse
import math
import joblib

class MatrixFactorization(object):

    def __init__(self, y_coo, num_factor, reg_bias, reg_factor, weight=None):
        if weight is None:
            weight = np.ones(y_coo.data.size)

        self.y_coo = y_coo
        self.y_csr = scipy.sparse.csr_matrix(y_coo)
        self.y_csc = scipy.sparse.csc_matrix(y_coo)
        self.num_factor = num_factor
        self.prior_param = {
            'col_bias_prec': reg_bias,
            'row_bias_prec': reg_bias,
            'factor_prec': reg_factor,
            'weight': weight,
            'df': 5.0,
        }

    def update_intercept(self, phi, mu_wo_intercept):
        post_prec = np.sum(phi)
        residual = self.y_coo.data - mu_wo_intercept
        post_mean = np.sum(phi * residual) / post_prec
        mu0 = np.random.normal(post_mean, 1 / math.sqrt(post_prec))
        return mu0

    def update_weight_param(self, mu0, r, u, c, v):
        # Returns the weight parameters in an 1-D array in the row major order
        # and also the mean estimate of matrix factorization as a by-product.

        prior_shape = self.prior_param['df'] / 2
        prior_rate = self.prior_param['df'] / 2 / self.prior_param['weight']

        i = self.y_coo.row
        j = self.y_coo.col

        mu = mu0 + r[i] + c[j] + np.sum(u[i, :] * v[j, :], 1)
        sq_error = (self.y_coo.data - mu) ** 2
        post_shape = prior_shape + 1 / 2
        post_rate = prior_rate + sq_error / 2
        phi = np.random.gamma(post_shape, 1 / post_rate)

        return phi, mu

    def update_row_param(self, phi_csr, mu0, c, v, r_prev, u_prev, num_process):

        nrow = self.y_csr.shape[0]
        num_factor = v.shape[1]

        # Update 'c' and 'v' block-wise in parallel.
        if num_process == 1:
            r, u = self.update_row_param_blockwise(self.y_csr, phi_csr, mu0, c, v, r_prev, u_prev)
        else:
            n_block = num_process
            block_ind = np.linspace(0, nrow, 1 + n_block, dtype=int)
            ru = joblib.Parallel(n_jobs=num_process)(
                joblib.delayed(self.update_row_param_blockwise)(self.y_csr[block_ind[m]:block_ind[m + 1], :],
                                                   phi_csr[block_ind[m]:block_ind[m + 1], :],
                                                   mu0, c, v,
                                                   r_prev[block_ind[m]:block_ind[m + 1]],
                                                   u_prev[block_ind[m]:block_ind[m + 1]])
                for m in range(n_block))
            r = np.concatenate([ru_i[0] for ru_i in ru])
            u = np.vstack([ru_i[1] for ru_i in ru])

        return r, u

    def update_row_param_blockwise(self, y_csr, phi_csr, mu0, c, v, r_prev, u_prev):

        nrow = y_csr.shape[0]
        prior_Phi = np.diag(np.hstack((self.prior_param['row_bias_prec'],
                                       np.tile(self.prior_param['factor_prec'], self.num_factor))))

        ru = [self.update_per_row(y_csr[i, :], phi_csr[i, :], mu0, c, v, r_prev[i], u_prev[i,:], prior_Phi) for i in range(nrow)]
        r = np.array([ru_i[0] for ru_i in ru])
        u = np.vstack([ru_i[1] for ru_i in ru])

        return r, u

    def update_per_row(self, y_csr, phi_csr, mu0, c, v, r_prev_i, u_prev_i, prior_Phi):

        J = y_csr.indices
        nnz_i = len(J)
        residual_i = y_csr.data - mu0 - c[J]
        phi_i = phi_csr.data
        v_T = np.hstack((np.ones((nnz_i, 1)), v[J, :]))
        post_Phi_i = prior_Phi + \
                     np.dot(v_T.T,
                            np.tile(phi_i[:, np.newaxis], (1, 1 + self.num_factor)) * v_T)  # Weighted sum of v_j * v_j.T
        post_mean_i = np.squeeze(np.dot(phi_i * residual_i, v_T))
        C, lower = scipy.linalg.cho_factor(post_Phi_i)
        post_mean_i = scipy.linalg.cho_solve((C, lower), post_mean_i)
        # Generate Gaussian, recycling the Cholesky factorization from the posterior mean computation.
        ru_i = math.sqrt(1 - self.relaxation ** 2) * scipy.linalg.solve_triangular(C, np.random.randn(len(post_mean_i)),
                                                                                   lower=lower)
        ru_i += post_mean_i + self.relaxation * (post_mean_i - np.concatenate(([r_prev_i], u_prev_i)))
        r_i = ru_i[0]
        u_i = ru_i[1:]

        return r_i, u_i

    def update_col_param(self, phi_csc, mu0, r, u, c_prev, v_prev, num_process):

        ncol = self.y_csc.shape[1]

        if num_process == 1:
            c, v = self.update_col_param_blockwise(self.y_csc, phi_csc, mu0, r, u, c_prev, v_prev)
        else:
            # Update 'c' and 'v' block-wise in parallel.
            n_block = num_process
            block_ind = np.linspace(0, ncol, 1 + n_block, dtype=int)
            cv = joblib.Parallel(n_jobs=num_process)(
                joblib.delayed(self.update_col_param_blockwise)(self.y_csc[:, block_ind[m]:block_ind[m + 1]],
                                                   phi_csc[:, block_ind[m]:block_ind[m + 1]],
                                                   mu0, r, u,
                                                   c_prev[block_ind[m]:block_ind[m + 1]],
                                                   v_prev[block_ind[m]:block_ind[m + 1]])
                for m in range(n_block))
            c = np.concatenate([cv_j[0] for cv_j in cv])
            v = np.vstack([cv_j[1] for cv_j in cv])

        return c, v

    def update_col_param_blockwise(self, y_csc, phi_csc, mu0, r, u, c_prev, v_prev):

        ncol = y_csc.shape[1]
        prior_Phi = np.diag(np.hstack((self.prior_param['col_bias_prec'],
                                       np.tile(self.prior_param['factor_prec'], self.num_factor))))

        cv = [self.update_per_col(y_csc[:, j], phi_csc[:, j], mu0, r, u, c_prev[j], v_prev[j,:], prior_Phi) for j in range(ncol)]
        c = np.array([cv_j[0] for cv_j in cv])
        v = np.vstack([cv_j[1] for cv_j in cv])

        return c, v

    def update_per_col(self, y_csc, phi_csc, mu0, r, u, c_prev_j, v_prev_j, prior_Phi):

        num_factor = u.shape[1]

        I = y_csc.indices
        nnz_j = len(I)
        residual_j = y_csc.data - mu0 - r[I]
        phi_j = phi_csc.data
        u_T = np.hstack((np.ones((nnz_j, 1)), u[I, :]))
        post_Phi_j = prior_Phi + \
                     np.dot(u_T.T,
                            np.tile(phi_j[:, np.newaxis], (1, 1 + num_factor)) * u_T)  # Weighted sum of u_i * u_i.T
        post_mean_j = np.squeeze(np.dot(phi_j * residual_j, u_T))
        C, lower = scipy.linalg.cho_factor(post_Phi_j)
        post_mean_j = scipy.linalg.cho_solve((C, lower), post_mean_j)
        # Generate Gaussian, recycling the Cholesky factorization from the posterior mean computation.
        cv_j = math.sqrt(1 - self.relaxation ** 2) * scipy.linalg.solve_triangular(C, np.random.randn(len(post_mean_j)),
                                                                              lower=lower)
        cv_j += post_mean_j + self.relaxation * (post_mean_j - np.concatenate(([c_prev_j], v_prev_j)))
        c_j = cv_j[0]
        v_j = cv_j[1:]

        return c_j, v_j

    def gibbs(self, n_burnin, n_mcmc, n_update=100, num_process=1, seed=None, relaxation=-0.0):

        np.random.seed(seed)
        self.relaxation = relaxation  # Recovers the standard Gibbs sampler when relaxation = 0.

        n_iter_per_update = max(1, math.floor((n_burnin + n_mcmc) / n_update))
        nrow, ncol = self.y_coo.shape

        # Pre-allocate
        logp_samples = np.zeros(n_burnin + n_mcmc)
        mu0_samples = np.zeros((n_mcmc, 1))
        c_samples = np.zeros((ncol, n_mcmc))
        v_samples = np.zeros((ncol, self.num_factor, n_mcmc))
        r_samples = np.zeros((nrow, n_mcmc))
        u_samples = np.zeros((nrow, self.num_factor, n_mcmc))
        post_mean_mu = np.zeros(self.y_coo.nnz)

        # Initial value
        r = np.zeros(nrow)
        u = np.zeros((nrow, self.num_factor))
        c = np.zeros(ncol)
        v = np.zeros((ncol, self.num_factor))
        phi = self.prior_param['weight']
        mu_wo_intercept = np.zeros(self.y_coo.nnz)

        # Gibbs steps
        for i in range(n_burnin + n_mcmc):

            mu0 = self.update_intercept(phi, mu_wo_intercept)
            phi_csr = scipy.sparse.csr_matrix((phi, (self.y_coo.row, self.y_coo.col)), self.y_coo.shape)
            r, u = self.update_row_param(phi_csr, mu0, c, v, r, u, num_process)
            phi_csc = scipy.sparse.csc_matrix((phi, (self.y_coo.row, self.y_coo.col)), self.y_coo.shape)
            c, v = self.update_col_param(phi_csc, mu0, r, u, c, v, num_process)
            phi, mu = self.update_weight_param(mu0, r, u, c, v)
            mu_wo_intercept = mu - mu0

            # Compute the log posterior (with the weight parameter marginalized out)
            logp_samples[i] = - (self.prior_param['df'] + 1) / 2 * np.sum(
                np.log(1 + (self.y_coo.data - mu) ** 2 * self.prior_param['weight'] / self.prior_param['df'])) + \
                              - self.prior_param['col_bias_prec'] / 2 * np.sum(c ** 2) + \
                              - self.prior_param['row_bias_prec'] / 2 * np.sum(v ** 2, (0, 1)) + \
                              - self.prior_param['factor_prec']  / 2 * np.sum(r ** 2) + \
                              - self.prior_param['factor_prec'] / 2 * np.sum(u ** 2, (0, 1))

            if i >= n_burnin:
                index = i - n_burnin
                mu0_samples[index] = mu0
                c_samples[:, index] = c
                u_samples[:, :, index] = u
                r_samples[:, index] = r
                v_samples[:, :, index] = v
                post_mean_mu = index / (index + 1) * post_mean_mu + 1 / (index + 1) * mu

            if ((i + 1) % n_iter_per_update) == 0:
                print('{:d} iterations have been completed.'.format(i + 1))
                print('The current log posterior is {:.3g}.'.format(logp_samples[i]))

        # Save outputs
        sample_dict = {
            'logp': logp_samples,
            'mu0': mu0_samples,
            'r': r_samples,
            'u': u_samples,
            'c': c_samples,
            'v': v_samples
        }

        return post_mean_mu, sample_dict

    # Old functions for row and column parameter updates. Saved in case it is easier to cythonize.
    def for_loop_update_row_param_blockwise(self, y_csr, phi_csr, mu0, c, v, r_prev, u_prev):

        nrow = y_csr.shape[0]
        num_factor = v.shape[1]
        prior_Phi = np.diag(np.hstack((self.prior_param['row_bias_prec'],
                                       np.tile(self.prior_param['factor_prec'], num_factor))))

        # Pre-allocate
        r = np.zeros(nrow)
        u = np.zeros((nrow, num_factor))

        # NOTE: The loop through 'i' is completely parallelizable.
        for i in range(nrow):
            j = y_csr[i, :].indices
            nnz_i = len(j)
            residual_i = y_csr[i, :].data - mu0 - c[j]
            phi_i = phi_csr[i, :].data.copy()

            v_T = np.hstack((np.ones((nnz_i, 1)), v[j, :]))
            post_Phi_i = prior_Phi + \
                         np.dot(v_T.T,
                                np.tile(phi_i[:, np.newaxis], (1, 1 + num_factor)) * v_T)  # Weighted sum of v_j * v_j.T
            post_mean_i = np.squeeze(np.dot(phi_i * residual_i, v_T))

            C, lower = scipy.linalg.cho_factor(post_Phi_i)
            post_mean_i = scipy.linalg.cho_solve((C, lower), post_mean_i)
            # Generate Gaussian, recycling the Cholesky factorization from the posterior mean computation.
            ru_i = math.sqrt(1 - self.relaxation ** 2) * scipy.linalg.solve_triangular(C, np.random.randn(len(post_mean_i)),
                                                                                       lower=lower)
            ru_i += post_mean_i + self.relaxation * (post_mean_i - np.concatenate(([r_prev[i]], u_prev[i, :])))
            r[i] = ru_i[0]
            u[i, :] = ru_i[1:]

        return r, u


    def for_loop_update_col_param_blockwise(self, y_csc, phi_csc, mu0, r, u, c_prev, v_prev):

        ncol = y_csc.shape[1]
        num_factor = u.shape[1]
        prior_Phi = np.diag(np.hstack((self.prior_param['col_bias_prec'],
                                       np.tile(self.prior_param['factor_prec'], num_factor))))

        # Pre-allocate
        c = np.zeros(ncol)
        v = np.zeros((ncol, num_factor))

        # NOTE: The loop through 'j' is completely parallelizable.
        for j in range(ncol):
            i = y_csc[:, j].indices
            nnz_j = len(i)
            residual_j = y_csc[:, j].data - mu0 - r[i]
            phi_j = phi_csc[:, j].data

            u_T = np.hstack((np.ones((nnz_j, 1)), u[i, :]))
            post_Phi_j = prior_Phi + \
                         np.dot(u_T.T,
                                np.tile(phi_j[:, np.newaxis], (1, 1 + num_factor)) * u_T)  # Weighted sum of u_i * u_i.T
            post_mean_j = np.squeeze(np.dot(phi_j * residual_j, u_T))

            C, lower = scipy.linalg.cho_factor(post_Phi_j)
            post_mean_j = scipy.linalg.cho_solve((C, lower), post_mean_j)
            # Generate Gaussian, recycling the Cholesky factorization from the posterior mean computation.
            cv_j = math.sqrt(1 - self.relaxation ** 2) * scipy.linalg.solve_triangular(C, np.random.randn(len(post_mean_j)),
                                                                                       lower=lower)
            cv_j += post_mean_j + self.relaxation * (post_mean_j - np.concatenate(([c_prev[j]], v_prev[j, :])))
            c[j] = cv_j[0]
            v[j, :] = cv_j[1:]

        return c, v