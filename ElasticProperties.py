#!/usr/bin/env python
# coding: utf-8

# Elastic Properties.py -  Extract elastic tensor and calculate mechanical properties from VASP OUTCAR file
# Equations can be found at https://www.materialsproject.org/wiki/index.php/Elasticity_calculations


import numpy as np


def get_elastic_tensor(filename):
    """
    Reads the elastic tensor from the OUTCAR.
    Args:
        filename : the name of the VASP OUTCAR
    Returns:
        elastic_tensor : 6x6 tensor of the elastic moduli
    """
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
    ii = 0
    copy = False
    elastic_tensor = []
    for line in lines:
        inp = line.split()
        if ii >= 6:
            continue
        if not inp:
            continue
        if len(inp) < 4 or len(inp) > 7:
            continue
        if len(inp) == 4 and inp[0] == "TOTAL":
            copy = True
        if copy:
            if len(inp) == 7 and len(inp[0]) == 2:
                elastic_tensor.append(inp[1:])
                ii += 1
    return np.asarray(elastic_tensor).astype(np.float)


def main():
    # Elastic tensor
    elastic_tensor = get_elastic_tensor("OUTCAR")
    c_ij = elastic_tensor / 10  # (GPa)

    # Compliance tensor (GPa)
    s_ij = np.linalg.inv(c_ij)

    # Voigt bulk modulus (GPa)
    k_v = (
        (c_ij[0, 0] + c_ij[1, 1] + c_ij[2, 2])
        + 2 * (c_ij[0, 1] + c_ij[1, 2] + c_ij[2, 0])
    ) / 9

    # Reuss bulk modulus (GPa)
    k_r = 1 / (
        (s_ij[0, 0] + s_ij[1, 1] + s_ij[2, 2])
        + 2 * (s_ij[0, 1] + s_ij[1, 2] + s_ij[2, 0])
    )

    # Voigt shear modulus (GPa)
    g_v = (
        (c_ij[0, 0] + c_ij[1, 1] + c_ij[2, 2])
        - (c_ij[0, 1] + c_ij[1, 2] + c_ij[2, 0])
        + 3 * (c_ij[3, 3] + c_ij[4, 4] + c_ij[5, 5])
    ) / 15

    # Reuss shear modulus (GPa)
    g_r = 15 / (
        4 * (s_ij[0, 0] + s_ij[1, 1] + s_ij[2, 2])
        - 4 * (s_ij[0, 1] + s_ij[1, 2] + s_ij[2, 0])
        + 3 * (s_ij[3, 3] + s_ij[4, 4] + s_ij[5, 5])
    )

    # Voigt-Reuss-Hill bulk modulus (GPa)
    k_vrh = (k_v + k_r) / 2

    # Voigt-Reuss-Hill shear modulus (GPa)
    g_vrh = (g_v + g_r) / 2

    # Universal elastic anisotropy
    au = 5 * (g_v / g_r) + (k_v / k_r) - 6
    if au < 0:
        au = 0

    # Isotropic Poisson ratio
    mu = (3 * k_vrh - 2 * g_vrh) / (6 * k_vrh + 2 * g_vrh)

    # Pugh's ratio
    eta = k_vrh / g_vrh

    # Pugh's hardness estimate (GPa)
    hardness_pugh = 2 * (g_vrh / eta ** 2) ** 0.585 - 3


if __name__ == "__main__":
    main()
