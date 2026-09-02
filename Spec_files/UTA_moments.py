#!/usr/bin/env python3
import numpy as np
import py3nj
import csv

# Will need to fix the ranks of the slater integrals as they rarely increase by orders of 1
def get_cowan_integrals(
    tsv_file,
    nconf,
    integral_type="FG",
    as_array=True,
):
    """
    Extract Cowan radial parameters for configuration number NCONF.

    Parameters
    ----------
    tsv_file : str
        Path to the Cowan .slater_labeled.tsv file.

    nconf : int
        Cowan configuration serial number.

    integral_type : str, optional
        Parameters to extract:

        "F"     -> F^k integrals only
        "G"     -> G^k integrals only
        "FG"    -> all F^k and G^k integrals
        "zeta"  -> spin-orbit parameters only
        "all"   -> F^k, G^k and zeta

        Default is "FG".

    as_array : bool, optional
        If True:
            return labels, values

        If False:
            return the complete matching rows.

    Returns
    -------
    labels : list[str]
        Parameter labels.

    values : numpy.ndarray
        Numerical values in the order they occur in the Cowan
        parameter list.
    """

    nconf = int(nconf)
    integral_type = integral_type.lower()

    valid_types = {"f", "g", "fg", "zeta", "all"}

    if integral_type not in valid_types:
        raise ValueError(
            "integral_type must be one of "
            "'F', 'G', 'FG', 'zeta', or 'all'"
        )

    rows = []

    with open(tsv_file, "r", newline="") as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            # Select configuration by Cowan NCONF
            if int(row["nconf"]) != nconf:
                continue

            parameter = row["parameter"].strip()
            plower = parameter.lower()

            # Determine parameter class
            is_F = parameter.startswith("F")
            is_G = parameter.startswith("G")
            is_zeta = plower.startswith("zeta")

            if integral_type == "f":
                keep = is_F

            elif integral_type == "g":
                keep = is_G

            elif integral_type == "fg":
                keep = is_F or is_G

            elif integral_type == "zeta":
                keep = is_zeta

            elif integral_type == "all":
                keep = is_F or is_G or is_zeta

            if keep:
                row["nconf"] = int(row["nconf"])
                row["index"] = int(row["index"])
                row["value"] = float(row["value"])
                row["JPAR"] = int(row["JPAR"])

                rows.append(row)

    if not rows:
        raise ValueError(
            f"No {integral_type.upper()} parameters found "
            f"for NCONF={nconf}"
        )

    if not as_array:
        return rows

    labels = [row["parameter"] for row in rows]

    values = np.array(
        [row["value"] for row in rows],
        dtype=float,
    )

    return labels, values

import re
import numpy as np


def get_integral_ranks(labels):
    """
    Extract the ranks k of Cowan F^k and G^k integrals
    from a list/array of parameter labels.

    Parameters
    ----------
    labels : sequence of str
        Integral labels such as:
        ["F2(5d,6d)", "F4(5d,6d)",
         "G0(5d,6d)", "G2(5d,6d)", "G4(5d,6d)"]

    Returns
    -------
    F_ranks : numpy.ndarray
        Ranks of all F^k integrals, in the same order
        that the F integrals occur in labels.

    G_ranks : numpy.ndarray
        Ranks of all G^k integrals, in the same order
        that the G integrals occur in labels.
    """

    F_ranks = []
    G_ranks = []

    for label in labels:

        label = label.strip()

        # F integral
        match_F = re.match(r"^F(\d+)\(", label)

        if match_F:
            F_ranks.append(int(match_F.group(1)))
            continue

        # G integral
        match_G = re.match(r"^G(\d+)\(", label)

        if match_G:
            G_ranks.append(int(match_G.group(1)))

    return (
        np.array(F_ranks, dtype=int),
        np.array(G_ranks, dtype=int),
    )

def D_1(F_k,F_ranks,n,l,N):
    #F_k is the array of the F^k slater integrals for a given configuration
    #n is the principle quantum number 
    #l is the orbital ang mom quantum number 
    #N is the number of electrons in the subshell
    #k=np.arange(1,len(F_k)+1)
    sum_val=0
    for ind_1,i in enumerate(F_ranks):
        for ind_2,j in enumerate(F_ranks):
            prefac_1= -1/((2*l+1)*(4*l+1))- py3nj.wigner6j(2*l,2*l,2*i,2*l,2*l,2*j)
            prefac_2=(2*l+1)**3/(8*l*((4*l)**2-1))*(py3nj.wigner3j(2*l,2*i,2*l,0,0,0)**2)*(py3nj.wigner3j(2*l,2*j,2*l,0,0,0)**2)
            prefac_3= N*(N-1)*(4*l-N+1)*(4*l-N+2)*F_k[ind_1]*F_k[ind_2]
            if i==j:
                prefac_1=prefac_1 + 2/(2*i+1)
            sum_val+=prefac_1*prefac_2*prefac_3
    return sum_val

def D_2(zeta,n,l,N):
    return (0.25*l*(l+1)/(4*l+1))*N*(4*l-N+2)*zeta**2

def D_3(F_k, F_ranks,n_1, n_2, l_1, l_2, N_1, N_2):
    # F_k is the array of direct Slater integrals F^k(n1 l1, n2 l2)
    # Table 3.2: sums over k != 0 and k' != 0

    #k = np.arange(1, len(F_k) + 1)

    b = N_1 * (4*l_1 - N_1 + 2) * N_2 * (4*l_2 - N_2 + 2)

    sum_val = 0

    for ind_1, i in enumerate(F_ranks):
        for ind_2, j in enumerate(F_ranks):

            # delta(k,k')
            delta = 1 if i == j else 0

            prefac_1 = (
                delta
                * (2*l_1 + 1)
                * (2*l_2 + 1)
                / ((2*i + 1) * (4*l_1 + 1) * (4*l_2 + 1))
            )

            prefac_2 = (
                py3nj.wigner3j(
                    2*l_1, 2*i, 2*l_1,
                    0, 0, 0
                )**2
                *
                py3nj.wigner3j(
                    2*l_2, 2*i, 2*l_2,
                    0, 0, 0
                )**2
            )

            prefac_3 = b * F_k[ind_1] * F_k[ind_2]

            sum_val += prefac_1 * prefac_2 * prefac_3

    return sum_val


def D_4(G_k, G_ranks, n_1, n_2, l_1, l_2, N_1, N_2):
    # G_k is the array of exchange Slater integrals G^k(n1 l1, n2 l2)

    #k = np.arange(1, len(G_k) + 1)

    b = N_1 * (4*l_1 - N_1 + 2) * N_2 * (4*l_2 - N_2 + 2)

    sum_val = 0

    for ind_1, i in enumerate(G_ranks):
        for ind_2, j in enumerate(G_ranks):

            delta = 1 if i == j else 0

            prefac_1 = (
                delta / (2*i + 1)
                - 1 / (4 * (2*l_1 + 1) * (2*l_2 + 1))
            )

            prefac_2 = (
                (2*l_1 + 1)
                * (2*l_2 + 1)
                / ((4*l_1 + 1) * (4*l_2 + 1))
            )

            prefac_3 = (
                py3nj.wigner3j(
                    2*l_1, 2*i, 2*l_2,
                    0, 0, 0
                )**2
                *
                py3nj.wigner3j(
                    2*l_1, 2*j, 2*l_2,
                    0, 0, 0
                )**2
            )

            prefac_4 = b * G_k[ind_1] * G_k[ind_2]

            sum_val += prefac_1 * prefac_2 * prefac_3 * prefac_4

    return sum_val


def D_5(F_k, G_k, F_ranks, G_ranks,n_1, n_2, l_1, l_2, N_1, N_2):
    # F_k = direct Slater integrals F^k(n1 l1, n2 l2)
    # G_k = exchange Slater integrals G^k(n1 l1, n2 l2)
    #
    # Table 3.2:
    #   sum over k != 0 for F^k
    #   sum over k' for G^k'

    #k_F = np.arange(1, len(F_k) + 1)
    #k_G = np.arange(1, len(G_k) + 1)

    b = N_1 * (4*l_1 - N_1 + 2) * N_2 * (4*l_2 - N_2 + 2)

    sum_val = 0

    for ind_F, i in enumerate(F_ranks):
        for ind_G, j in enumerate(G_ranks):

            prefac_1 = (-1)**(j + 1)

            prefac_2 = py3nj.wigner6j(
                2*l_2, 2*l_2, 2*i,
                2*l_1, 2*l_1, 2*j
            )

            prefac_3 = (
                (2*l_1 + 1)
                * (2*l_2 + 1)
                / ((4*l_1 + 1) * (4*l_2 + 1))
            )

            prefac_4 = (
                py3nj.wigner3j(
                    2*l_1, 2*i, 2*l_1,
                    0, 0, 0
                )
                *
                py3nj.wigner3j(
                    2*l_2, 2*i, 2*l_2,
                    0, 0, 0
                )
                *
                py3nj.wigner3j(
                    2*l_1, 2*j, 2*l_2,
                    0, 0, 0
                )**2
            )

            prefac_5 = b * F_k[ind_F] * G_k[ind_G]

            sum_val += (
                prefac_1
                * prefac_2
                * prefac_3
                * prefac_4
                * prefac_5
            )

    return sum_val
    

