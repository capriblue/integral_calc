import argparse
import os
import pickle

import numpy as np
import vegas
from scipy.special import j0, j1


def _r(r_a, theta_a, r_b,  theta_b):
    return np.sqrt(r_a**2 + r_b**2 - 2 * r_a * r_b * np.cos(theta_a - theta_b))


def hat_r(r_a, theta_a, r_b, theta_b, r_ab):
    return np.divide(np.array([r_a * np.cos(theta_a) - r_b * np.cos(theta_b), r_a * np.sin(theta_a) - r_b * np.sin(theta_b), np.zeros((r_a.shape[0]))]), r_ab, out=np.zeros((3, r_a.shape[0])), where=r_ab != 0)


def n(r_a, theta_a, kF_lambda, q, phi_0):
    denominator = r_a**2 + kF_lambda**2
    return np.divide(np.array([2*(kF_lambda * r_a * np.cos(q * theta_a - phi_0)), 2*(kF_lambda * r_a * np.sin(q * theta_a - phi_0)), r_a**2 - kF_lambda**2]), denominator, out=np.zeros((3, r_a.shape[0])), where=denominator != 0)


class batch_integrant(vegas.RBatchIntegrand):
    def __init__(self, q, kF_lambda, phi_0):
        self.q = q
        self.kF_lambda = kF_lambda
        self.phi_0 = phi_0

    def __call__(self, x):
        r_h, theta_h, r_i, theta_i, r_j, theta_j = x
        return self.__integrant(r_h, theta_h, r_i, theta_i, r_j, theta_j, self.q, self.kF_lambda, self.phi_0)

    def __integrant(self, r_h, theta_h, r_i, theta_i, r_j, theta_j, q, kF_lambda, phi_0):
        r_ij = _r(r_i, theta_i, r_j, theta_j)
        r_ih = _r(r_i, theta_i, r_h, theta_h)
        r_jh = _r(r_j, theta_j, r_h, theta_h)
        n_i = n(r_i, theta_i, kF_lambda, q, phi_0)
        n_j = n(r_j, theta_j, kF_lambda, q, phi_0)
        n_h = n(r_h, theta_h, kF_lambda, q, phi_0)
        hat_r_ih = hat_r(r_i, theta_i, r_h, theta_h, r_ih)
        hat_r_jh = hat_r(r_j, theta_j, r_h, theta_h, r_jh)
        integrant = (
            j0(r_ij) * j1(r_ih) * j1(r_jh) *
            r_h * r_i * r_j *
            (hat_r_ih[0] * hat_r_jh[1] - hat_r_ih[1] * hat_r_jh[0]) *
            np.sum(n_i * np.cross(n_j.T, n_h.T).T, axis=0)
        )
        return integrant


def get_result(q, phi_0, kF_lambda, filename,  k_FL, nitn, neval):
    integrant = batch_integrant(q, kF_lambda, phi_0)
    integrator = vegas.Integrator(
        [[0, k_FL], [0, 2*np.pi], [0, k_FL], [0, 2*np.pi], [0, k_FL], [0, 2*np.pi]])
    __warmup = integrator(integrant, nitn=nitn, neval=neval)
    result = integrator(integrant, nitn=nitn, neval=neval,
                        adapt=False, saveall=filename)
    return result


def continue_exec(q, phi_0, kF_lambda, filename,  k_FL, nitn, neval):
    integrant = batch_integrant(q, kF_lambda, phi_0)
    with open(filename, "rb") as f:
        old_result, integ = pickle.load(f)
    result = vegas.ravg(old_result.itn_results[5:])
    __warmup = integ(integrant, nitn=nitn, neval=neval)
    new_result = integ(integrant, nitn=nitn, neval=neval,
                       adapt=False, saveall=filename)
    print("===================================")
    print(f"old result: ")
    print(old_result.summary())
    print("===================================")
    result.extend(new_result)
    return result


def main():
    ps = argparse.ArgumentParser(
        description="numerical integral code using vegas algorithm")
    ps.add_argument("kF_lambda", type=float, help="kF_lambda")
    ps.add_argument("--q", type=float, help="q=1.0", default=1)
    ps.add_argument("--phi_0", type=float, help="phi_0=0.0", default=0)
    ps.add_argument("--k_FL", type=float, help="k_FL=10.0", default=10)
    ps.add_argument("--nitn", type=int,
                    help="nitn=20 > 5 (expected)", default=20)
    ps.add_argument("--neval", type=float,
                    help="neval=1e+4 > 1 (expected)", default=1e4)
    ps.add_argument("--fn", type=str, help="default: kF_lambda(kF_lambda:3e)_q(q:.1f)_phi_0(phi_0:.3f)_k_FL(k_FL:.2f).pkl",
                    default=None)
    ps.add_argument("--path", type=str,
                    help="path to save the file", default=".")
    args = ps.parse_args()
    if args.fn is None:
        args.fn = f"kF_lambda{args.kF_lambda:.3e}_q{args.q:.1f}_phi_0{args.phi_0:.3f}_k_FL{args.k_FL:.2f}.pkl"
    file_path = os.path.join(args.path, args.fn)
    if os.path.exists(file_path):
        print(f"file_path is already exists: continue execution mode")
        r = continue_exec(args.q, args.phi_0, args.kF_lambda,
                          file_path, args.k_FL, args.nitn, args.neval)
    else:
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        r = get_result(args.q, args.phi_0, args.kF_lambda,
                       file_path, args.k_FL, args.nitn, args.neval)
    print(r.summary())
    print(f"saved file as {args.fn}")
    print(
        f"q: {args.q}, phi_0: {args.phi_0}, kF_lambda: {args.kF_lambda}, k_FL: {args.k_FL}, nitn: {args.nitn}, nevanl: {args.neval} \n  {r.mean} ± {r.sdev} Q: {r.Q}")


if __name__ == "__main__":
    main()
