#!/usr/bin/env bash
# Download FLUX-family pretrained weights into ${HF_HOME}.
#
# Usage:
#   scripts/download_pretrained_weights.sh <target>
#   scripts/download_pretrained_weights.sh <target> --dry-run
#
# Targets:
#   repae                    REPA-E/e2e-flux-vae (~2 GB; needed for ORION
#                            and any REPA-E ablation)
#   flux1-kontext-dev        black-forest-labs/FLUX.1-Kontext-dev (~37 GB;
#                            already present on this cluster)
#   flux2-klein-4b-base      black-forest-labs/FLUX.2-klein-4B-base (~8 GB;
#                            Apache 2.0, recommended for FLUX.2 LoRA/FT)
#   flux2-klein-9b-base      black-forest-labs/FLUX.2-klein-9B-base (~18 GB;
#                            non-commercial)
#   flux2-dev                black-forest-labs/FLUX.2-dev (~64 GB; non-commercial)
#   all                      every target above
#
# Honours HF_HOME from the environment (default: ${HOME}/.cache/huggingface).
# Uses --resume-download so interrupted transfers can be re-tried.

set -euo pipefail

DRY_RUN=0
TARGET=""

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,22p' "${BASH_SOURCE[0]}"  # print the usage block above
      exit 0
      ;;
    *)
      if [[ -z "${TARGET}" ]]; then
        TARGET="${arg}"
      else
        echo "Unexpected extra argument: ${arg}" >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -z "${TARGET}" ]]; then
  echo "Missing target. Use --help to see options." >&2
  exit 2
fi

# Repo IDs keyed by short target name.
declare -A REPO_IDS=(
  [repae]="REPA-E/e2e-flux-vae"
  [flux1-kontext-dev]="black-forest-labs/FLUX.1-Kontext-dev"
  [flux2-klein-4b-base]="black-forest-labs/FLUX.2-klein-4B-base"
  [flux2-klein-9b-base]="black-forest-labs/FLUX.2-klein-9B-base"
  [flux2-dev]="black-forest-labs/FLUX.2-dev"
)

# Helper to run (or print, in --dry-run) a download for a single target.
download_one() {
  local short="$1"
  local repo="${REPO_IDS[${short}]:-}"
  if [[ -z "${repo}" ]]; then
    echo "Unknown target: ${short}" >&2
    exit 2
  fi
  local cmd=(
    huggingface-cli download
    "${repo}"
    --resume-download
  )
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "[dry-run] HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}  ${cmd[*]}"
  else
    echo "→ downloading ${repo} into ${HF_HOME:-${HOME}/.cache/huggingface}"
    "${cmd[@]}"
  fi
}

if [[ "${TARGET}" == "all" ]]; then
  for k in repae flux1-kontext-dev flux2-klein-4b-base flux2-klein-9b-base flux2-dev; do
    download_one "${k}"
  done
else
  download_one "${TARGET}"
fi

echo "Done."
