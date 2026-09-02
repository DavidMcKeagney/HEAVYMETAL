#!/usr/bin/env bash
# extract_cowan_slater_v3.sh
#
# Extract and fully label Cowan RCN single-configuration energy parameters
# from an OUT36 file.
#
# Usage:
#   ./extract_cowan_slater_v3.sh Hf_I.out36
#
# Outputs:
#   Hf_I.out36.slater_labeled.tsv
#   Hf_I.out36.zeta_bw.tsv
#
# The label reconstruction follows Cowan's documented RCG parameter order:
#   Eav
#   F^k(li,li) for each open subshell
#   zeta(li) for each open subshell with l > 0
#   F^k(li,lj) for each pair of open subshells
#   G^k(li,lj) for each pair of open subshells
#
# Here "open" means 1 <= w <= 4*l+1.  A completely filled subshell
# (w = 4*l+2) contributes no term-dependent Slater/spin-orbit parameters.
#
# Compact OUT36 parameter records store VALUE,JPAR pairs, where:
#   0 Eav
#   1 F^k(i,i)
#   2 zeta(i)
#   3 F^k(i,j)
#   4 G^k(i,j)
#   5 R^k  (not expected in the RCN single-configuration record)
#
# Eav is printed in rydbergs; the other compact parameters are in kK
# (= 1000 cm^-1).

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 OUT36-file" >&2
    exit 2
fi

infile=$1
if [[ ! -r "$infile" ]]; then
    echo "Error: cannot read '$infile'" >&2
    exit 1
fi

out="${infile}.slater_labeled.tsv"
zout="${infile}.zeta_bw.tsv"

awk -v OUT="$out" -v ZOUT="$zout" '
BEGIN {
    OFS="\t"
    print "element","configuration","nconf","parameter","value","units","JPAR" > OUT
    print "element","configuration","nconf","orbital","occupation","zeta_BW_Ry","zeta_BW_cm-1" > ZOUT

    in_orb=0; orb_started=0; in_zeta=0; collecting=0
}

function lvalue(orb, c) {
    c=substr(orb,length(orb),1)
    if (c=="s") return 0
    if (c=="p") return 1
    if (c=="d") return 2
    if (c=="f") return 3
    if (c=="g") return 4
    if (c=="h") return 5
    if (c=="i") return 6
    return -1
}

function clear_orbs( i) {
    for (i=1;i<=norb;i++) {
        delete orb[i]; delete occ[i]; delete ell[i]
    }
    norb=0
}

function clear_params( i) {
    for (i=1;i<=nval;i++) { delete pval[i]; delete pcode[i] }
    nval=0
}

function clear_labels( i) {
    for (i=1;i<=nlab;i++) { delete lab[i]; delete labcode[i] }
    nlab=0
}

function add_label(name,code) {
    nlab++
    lab[nlab]=name
    labcode[nlab]=code
}

function is_active(i) {
    # Cowan condition for nonequivalent F/G and zeta parameters:
    # 1 <= w <= 4*l+1.  Thus a full shell w=4*l+2 is excluded.
    return (occ[i] >= 1 && occ[i] <= 4*ell[i]+1)
}

function build_labels( i,j,k,kmax) {
    clear_labels()
    add_label("Eav",0)

    # Equivalent-electron F^k(li,li): F2,F4,...,F^(2l)
    for (i=1;i<=norb;i++) {
        if (!is_active(i)) continue
        if (occ[i] >= 2 && occ[i] <= 4*ell[i]) {
            for (k=2;k<=2*ell[i];k+=2)
                add_label("F" k "(" orb[i] "," orb[i] ")",1)
        }
    }

    # Spin-orbit parameters, in orbital order.
    for (i=1;i<=norb;i++) {
        if (is_active(i) && ell[i] > 0)
            add_label("zeta(" orb[i] ")",2)
    }

    # Nonequivalent direct F^k(li,lj): F2,F4,... min(2li,2lj)
    for (i=1;i<=norb;i++) {
        if (!is_active(i)) continue
        for (j=i+1;j<=norb;j++) {
            if (!is_active(j)) continue
            if (ell[i] > 0 && ell[j] > 0) {
                kmax=(2*ell[i] < 2*ell[j] ? 2*ell[i] : 2*ell[j])
                for (k=2;k<=kmax;k+=2)
                    add_label("F" k "(" orb[i] "," orb[j] ")",3)
            }
        }
    }

    # Nonequivalent exchange G^k(li,lj):
    # k=|li-lj|, |li-lj|+2, ... li+lj.
    # G0 is therefore retained for different-n orbitals having the same l.
    for (i=1;i<=norb;i++) {
        if (!is_active(i)) continue
        for (j=i+1;j<=norb;j++) {
            if (!is_active(j)) continue
            k=ell[i]-ell[j]; if (k<0) k=-k
            for (; k<=ell[i]+ell[j]; k+=2)
                add_label("G" k "(" orb[i] "," orb[j] ")",4)
        }
    }
}

function add_pairs(first, i) {
    for (i=first;i+1<=NF;i+=2) {
        if ($i ~ /^[-+]?[0-9]*\.?[0-9]+([EeDd][-+]?[0-9]+)?$/ &&
            $(i+1) ~ /^[0-9]+$/) {
            nval++
            pval[nval]=$i
            pcode[nval]=$(i+1)+0
        }
    }
}

function flush_params( i,u,nm) {
    build_labels()

    if (nlab != declared_npar) {
        print "WARNING: configuration " element " " config \
              ": reconstructed " nlab " labels but OUT36 declares " \
              declared_npar " parameters." > "/dev/stderr"
    }
    if (nval != declared_npar) {
        print "WARNING: configuration " element " " config \
              ": read " nval " VALUE,JPAR pairs but OUT36 declares " \
              declared_npar "." > "/dev/stderr"
    }

    for (i=1;i<=nval;i++) {
        nm=(i<=nlab ? lab[i] : "UNKNOWN")
        u=(pcode[i]==0 ? "Ry" : "kK")
        print element,config,nconf,nm,pval[i],u,pcode[i] > OUT

        if (i<=nlab && pcode[i] != labcode[i]) {
            print "WARNING: " element " " config ": " nm \
                  " reconstructed as JPAR=" labcode[i] \
                  " but OUT36 stores JPAR=" pcode[i] > "/dev/stderr"
        }
    }
    clear_params()
    collecting=0
}

# Start of a new RCN configuration block.
/^[[:space:]][[:space:]][[:space:]][^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+nconf=/ {
    element=$1
    config=$2
    nconf=0
    for (i=1;i<=NF;i++) {
        if ($i=="nconf=") { nconf=$(i+1)+0; break }
        if ($i ~ /^nconf=/) {
            split($i,a,"="); nconf=a[2]+0
            if (nconf==0 && i<NF) nconf=$(i+1)+0
            break
        }
    }
    clear_orbs()
    in_orb=0; orb_started=0; in_zeta=0
    next
}

# First radial-orbital table supplies the actual occupation numbers.
/^[[:space:]]*nl[[:space:]]+wnl[[:space:]]+ee[[:space:]]+/ {
    in_orb=1
    orb_started=0
    next
}

in_orb {
    if ($1 ~ /^[0-9]+[spdfghi]$/ && $2 ~ /^[0-9]+\.?$/) {
        norb++
        orb[norb]=$1
        occ[norb]=int($2+0)
        ell[norb]=lvalue($1)
        orb_started=1
        next
    }
    if (orb_started && NF>0) in_orb=0
}

# Blume-Watson zeta table: copy the useful orbital-resolved values too.
/index_dummy_never_matches/ { }

/--------------------zeta---------------------/ {
    in_zeta=1
    next
}

in_zeta {
    if ($1 ~ /^[0-9]+[spdfghi]$/ && $2 ~ /^[0-9]+\.?$/ && NF>=4) {
        print element,config,nconf,$1,$2,$3,$4 > ZOUT
        next
    }
    if ($0 ~ /^1[^[:space:]]/ || $0 ~ /^[[:space:]]*0[[:space:]]/) {
        in_zeta=0
    }
}

# Compact RCN parameter record.  Its first three fields are
# element, configuration, and number of stored parameters.
! /nconf=/ && $1==element && $2==config && $3 ~ /^[0-9]+$/ {
    declared_npar=$3+0
    clear_params()
    collecting=1
    add_pairs(4)
    if (nval >= declared_npar) flush_params()
    next
}

# Long parameter records continue on one or more numeric-only lines.
collecting {
    # A continuation line consists solely of VALUE,JPAR pairs.
    if (NF>=2 && $1 ~ /^[-+]?[0-9]*\.?[0-9]+([EeDd][-+]?[0-9]+)?$/ && $2 ~ /^[0-9]+$/) {
        add_pairs(1)
        if (nval >= declared_npar) flush_params()
        next
    }
}

END {
    if (collecting && nval>0) flush_params()
}
' "$infile"

echo "Wrote:"
echo "  $out"
echo "  $zout"
