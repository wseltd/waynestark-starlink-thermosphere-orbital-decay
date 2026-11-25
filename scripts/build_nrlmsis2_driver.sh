#!/usr/bin/env bash
# Build the WSE NRLMSIS 2.x line-protocol driver against the OFFICIAL sources.
#
# Copyright 2026 Waynestark Enterprises Limited. Apache-2.0 (this script and
# the driver source only; the NRL model sources it compiles are licensed by
# NRL under their own terms and are never modified, copied or redistributed).
#
# Usage:
#   scripts/build_nrlmsis2_driver.sh 2.0
#   scripts/build_nrlmsis2_driver.sh 2.1
#
# The official package location comes from NRLMSIS20_ROOT / NRLMSIS21_ROOT
# (names documented in .env.example; each user installs the package
# separately from the authorised NRL source, OUTSIDE this repository).
#
# The script refuses to build unless:
#   * the environment variable for the requested version is set and the
#     directory exists;
#   * the root lies OUTSIDE this repository tree (the licence forbids
#     bringing NRL material into any distributed artefact, so it may never
#     even sit inside the working tree);
#   * the sha256 of EVERY required file matches the public pin in
#     third_party/nrlmsis_asset_manifest.yaml. The two .parm files are the
#     same size and msisinit performs no content validation, so this check
#     is the only thing standing between a swapped or corrupted asset and a
#     silently wrong model.
#
# Build flags are EXACTLY the pre-registered ones from
# config/nrlmsis2_reference_tolerances.yaml:
#     2.1:  -O3 -cpp -DDBLE
#     2.0:  -O3 -cpp -DDBLE -std=legacy
# (-std=legacy is the recorded amendment for gfortran 11's bounds diagnostic
# on the 2.0 package's bspline call; it changes no NRL source line.)
# The compile is a single gfortran invocation: no make, no parallel jobs.
#
# Output: $ROOT/build/wse_msis2_driver — inside the EXTERNAL installation,
# never inside the repository.

set -euo pipefail

usage() {
    echo "usage: $0 <2.0|2.1>" >&2
    exit 2
}

[ $# -eq 1 ] || usage
VERSION="$1"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
MANIFEST="${REPO_ROOT}/third_party/nrlmsis_asset_manifest.yaml"
DRIVER_SRC="${REPO_ROOT}/src/thermosphere_consequence/providers/fortran/wse_msis2_driver.F90"

case "${VERSION}" in
    2.0)
        ENV_NAME="NRLMSIS20_ROOT"
        LOGICAL_ID="nrlmsis_2_0_package"
        FLAGS=(-O3 -cpp -DDBLE -std=legacy)
        # Canonical source order from the package readme, test program excluded.
        SOURCES=(alt2gph.F90 msis_constants.F90 msis_init.F90 msis_gfn.F90
                 msis_tfn.F90 msis_dfn.F90 msis_calc.F90 msis_gtd8d.F90)
        ;;
    2.1)
        ENV_NAME="NRLMSIS21_ROOT"
        LOGICAL_ID="nrlmsis_2_1_package"
        FLAGS=(-O3 -cpp -DDBLE)
        SOURCES=(msis_constants.F90 msis_utils.F90 msis_init.F90 msis_gfn.F90
                 msis_tfn.F90 msis_dfn.F90 msis_calc.F90 msis_gtd8d.F90)
        ;;
    *)
        usage
        ;;
esac

ROOT="${!ENV_NAME:-}"
if [ -z "${ROOT}" ]; then
    echo "error: ${ENV_NAME} is not set." >&2
    echo "Install the official NRLMSIS ${VERSION} package from the authorised NRL" >&2
    echo "source (see third_party/nrlmsis_asset_manifest.yaml) OUTSIDE this" >&2
    echo "repository, then export ${ENV_NAME}=<installation directory>." >&2
    exit 1
fi
if [ ! -d "${ROOT}" ]; then
    echo "error: ${ENV_NAME}=${ROOT} is not a directory." >&2
    exit 1
fi
ROOT="$(cd -- "${ROOT}" && pwd -P)"

case "${ROOT}/" in
    "${REPO_ROOT}/"*)
        echo "error: ${ENV_NAME}=${ROOT} lies INSIDE this repository (${REPO_ROOT})." >&2
        echo "The NRL licence forbids bringing the official sources or coefficient" >&2
        echo "files into any distributed artefact; install them outside the tree." >&2
        exit 1
        ;;
esac

[ -f "${MANIFEST}" ] || { echo "error: manifest not found at ${MANIFEST}" >&2; exit 1; }
[ -f "${DRIVER_SRC}" ] || { echo "error: driver source not found at ${DRIVER_SRC}" >&2; exit 1; }

# Extract "<name> <sha256>" pairs for this version's required_files block.
# The manifest is committed YAML with one `- {name: ..., sha256: ...}` entry
# per line; scoping stops at the next `- logical_id:` block.
required_files() {
    awk -v id="${LOGICAL_ID}" '
        /- logical_id:/ { inblock = ($0 ~ id) ; next }
        inblock && /- \{name:/ {
            line = $0
            sub(/.*name:[ \t]*/, "", line)
            name = line
            sub(/,.*/, "", name)
            sha = line
            sub(/.*sha256:[ \t]*/, "", sha)
            sub(/[^0-9a-f].*$/, "", sha)
            print name, sha
        }
    ' "${MANIFEST}"
}

PAIRS="$(required_files)"
if [ -z "${PAIRS}" ]; then
    echo "error: no required_files found for ${LOGICAL_ID} in ${MANIFEST}" >&2
    exit 1
fi

echo "Verifying official NRLMSIS ${VERSION} assets in ${ROOT} against ${MANIFEST}"
FAILED=0
while read -r NAME SHA; do
    FILE="${ROOT}/${NAME}"
    if [ ! -f "${FILE}" ]; then
        echo "  MISSING  ${NAME}" >&2
        FAILED=1
        continue
    fi
    ACTUAL="$(sha256sum "${FILE}" | cut -d' ' -f1)"
    if [ "${ACTUAL}" != "${SHA}" ]; then
        echo "  MISMATCH ${NAME}: expected ${SHA}, got ${ACTUAL}" >&2
        FAILED=1
    else
        echo "  ok       ${NAME}"
    fi
done <<< "${PAIRS}"
if [ "${FAILED}" -ne 0 ]; then
    echo "error: asset verification FAILED; refusing to build. Re-install the" >&2
    echo "official package from the authorised source named in ${MANIFEST}." >&2
    exit 1
fi

BUILD_DIR="${ROOT}/build"
mkdir -p "${BUILD_DIR}"
OUT="${BUILD_DIR}/wse_msis2_driver"

SOURCE_PATHS=()
for SRC in "${SOURCES[@]}"; do
    SOURCE_PATHS+=("${ROOT}/${SRC}")
done

# One sequential gfortran invocation; -J keeps the .mod files in the build
# directory of the EXTERNAL root, so nothing lands in the repository.
CMD=(gfortran "${FLAGS[@]}" -J "${BUILD_DIR}" -o "${OUT}"
     "${SOURCE_PATHS[@]}" "${DRIVER_SRC}")
echo "Compile command:"
printf '  %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "Built ${OUT}"
