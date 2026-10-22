#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

data_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
archives=()
destinations=()

for group in pusht/rollouts pusht/masks reacher fetch_slide beerpong analysis/reacher_extrapolation analysis/pusht_context/rollouts; do
  destination="$data_dir/$group"
  if [[ "$group" == pusht/* ]]; then
    destination="$data_dir/pusht/fresh_iid/${group#pusht/}"
  fi
  for archive in "$data_dir/tar/$group/"*.tar; do
    name="${archive##*/}"
    output="$destination/${name%.tar}.zarr"
    if [[ -e "$output" || -L "$output" ]]; then
      printf 'Refusing to overwrite: %s\n' "$output" >&2
      exit 1
    fi
    archives+=("$archive")
    destinations+=("$destination")
  done
done

if (( ${#archives[@]} == 0 )); then
  printf 'No .tar files found under %s/tar/\n' "$data_dir" >&2
  exit 1
fi

for index in "${!archives[@]}"; do
  archive="${archives[index]}"
  destination="${destinations[index]}"
  mkdir -p -- "$destination"
  printf 'Extracting %s\n' "$archive"
  tar --extract --file="$archive" --directory="$destination" \
    --keep-old-files --no-same-owner --no-same-permissions
done

printf 'Extracted %s archives. Original .tar files were retained.\n' "${#archives[@]}"
