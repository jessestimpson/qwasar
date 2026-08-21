#!/bin/sh
# qwasar model downloader.
#
# Fetches the weights qwasar needs and links them where the binaries look by
# default, so that after
#
#     ./download_model.sh model
#
# this works from the project directory with no arguments:
#
#     ./qwasar -p "Hello"
#
# Only the six files qwasar actually reads are downloaded -- config.json, the
# safetensors index, the three shards, and tokenizer.json.  The rest of the
# repository is a chat template qwasar embeds verbatim (see THIRD-PARTY.md) and
# preprocessor configs for the vision tower, which is not implemented.
#
# Downloads resume: run the same command again after an interruption.
set -e

MODEL_REPO="lmstudio-community/Qwen3.8-27B-MLX-4bit"
MODEL_REV="6067b15cf581666a4aecf6af3afaba4bb5efc20c"
MODEL_NAME="Qwen3.8-27B-MLX-4bit"
MODEL_FILES="config.json
model.safetensors.index.json
tokenizer.json
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors"

MTP_REPO="EigenLabs/Qwen3.8-27B-MTP-bf16"
MTP_REV="26a328e070875b0314d652a039b6b59902690f03"
MTP_NAME="Qwen3.8-27B-MTP-bf16"
MTP_FILES="config.json
model.safetensors.index.json
model.safetensors"

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
OUT_DIR=${QWASAR_MODEL_DIR:-"$ROOT/models"}
case "$OUT_DIR" in
    /*) ;;
    *) OUT_DIR="$ROOT/$OUT_DIR" ;;
esac
TOKEN=${HF_TOKEN:-}
VERIFY=0

usage() {
    cat <<EOF
qwasar model downloader

Usage:
  ./download_model.sh model    [--verify] [--token TOKEN]
  ./download_model.sh mtp-head [--verify] [--token TOKEN]
  ./download_model.sh all      [--verify] [--token TOKEN]

Targets:

  model
       Qwen3.8 27B, MLX affine 4-bit, group 64.  About 16 GB on disk.
       This is the only quantisation qwasar can read: the kernels are built
       around that one format on purpose (PLAN.md section 1.2), so an 8-bit or
       6-bit conversion of the same model will not load.

       Repository: $MODEL_REPO
       Links ./qwasar-model, which every qwasar binary uses when -m is absent.

  mtp-head
       The multi-token-prediction draft head, bf16.  About 850 MB.
       Optional, and not useful yet: speculative decoding is milestone 3 and
       is still being built.  The head ships separately from the checkpoint
       because merging it breaks Python loaders -- see PLAN.md.

       Repository: $MTP_REPO
       Links ./qwasar-mtp, which is what to pass to --mtp.

  all
       Both of the above.

Options:
  --verify       Check the SHA-256 of every downloaded file.  Sizes are always
                 checked; this additionally re-reads 16 GB, so it is opt-in.
  --token TOKEN  Hugging Face token.  Neither repository needs one; HF_TOKEN
                 and the local token cache are used if present.

Environment:
  QWASAR_MODEL_DIR  Where downloads are kept.  Default: ./models
  QWASAR_MODEL      Checked by the binaries before ./qwasar-model, so an
                    existing copy elsewhere can be used without downloading.

Both repositories are pinned to a revision, so a re-run fetches the same bytes
this was tested against rather than whatever main has become.
EOF
}

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

TARGET=$1
shift
case "$TARGET" in
    model|mtp-head|all) ;;
    -h|--help|help) usage; exit 0 ;;
    *)
        echo "Unknown target: $TARGET" >&2
        echo >&2
        usage >&2
        exit 1
        ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            shift
            [ $# -gt 0 ] || { echo "Missing value after --token" >&2; exit 1; }
            TOKEN=$1
            ;;
        --verify) VERIFY=1 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
    shift
done

if [ -z "$TOKEN" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi

# Asking the server for the expected size and digest rather than pinning them
# here means the check cannot go stale against the pinned revision.
#
# Redirects have to be followed to learn either: a HEAD on the repository URL
# answers with the length of the redirect body, not of the file, which is a
# trap worth naming because the number it returns looks perfectly plausible.
# The LAST content-length in the chain is the real one.
#
# x-linked-etag is a SHA-256 for the LFS files and a git blob SHA-1 for the
# small ones, so only the 64-character form is usable as a digest.
remote_meta() {
    if [ -n "$TOKEN" ]; then
        curl -sIL -H "Authorization: Bearer $TOKEN" --max-time 120 "$1"
    else
        curl -sIL --max-time 120 "$1"
    fi | tr -d '\r' | awk 'BEGIN{IGNORECASE=1}
             /^content-length:/{s=$2}
             /^x-linked-etag:/{gsub(/"/,"",$2); if (length($2)==64) e=$2}
             END{print s, e}'
}

file_size() {
    wc -c < "$1" | tr -d ' '
}

download_one() {
    repo=$1; rev=$2; file=$3; dir=$4
    out="$dir/$file"
    part="$out.part"
    url="https://huggingface.co/$repo/resolve/$rev/$file"

    meta=$(remote_meta "$url")
    want_size=$(printf '%s' "$meta" | cut -d' ' -f1)
    want_sha=$(printf '%s' "$meta" | cut -d' ' -f2)

    if [ -s "$out" ]; then
        if [ -n "$want_size" ] && [ "$(file_size "$out")" != "$want_size" ]; then
            echo "Size mismatch, re-downloading: $out" >&2
            rm -f "$out"
        else
            echo "Already downloaded: $file"
            [ "$VERIFY" -eq 1 ] || return 0
        fi
    fi

    if [ ! -s "$out" ]; then
        echo "Downloading $file from $repo"
        if [ -n "$TOKEN" ]; then
            curl -fL --progress-meter -C - -H "Authorization: Bearer $TOKEN" -o "$part" "$url"
        else
            curl -fL --progress-meter -C - -o "$part" "$url"
        fi
        mv "$part" "$out"

        # A truncated download is the failure this catches, and it is worth
        # catching here: qwasar would otherwise report it as a malformed
        # safetensors header several gigabytes into a load.
        if [ -n "$want_size" ] && [ "$(file_size "$out")" != "$want_size" ]; then
            echo "Downloaded $out is $(file_size "$out") bytes, expected $want_size." >&2
            echo "Run this command again to resume." >&2
            exit 1
        fi
    fi

    if [ "$VERIFY" -eq 1 ] && [ -n "$want_sha" ]; then
        got=$(shasum -a 256 "$out" | cut -d' ' -f1)
        if [ "$got" != "$want_sha" ]; then
            echo "SHA-256 mismatch for $out" >&2
            echo "  expected $want_sha" >&2
            echo "  got      $got" >&2
            exit 1
        fi
        echo "  verified $file"
    fi
}

fetch() {
    repo=$1; rev=$2; name=$3; files=$4; link=$5
    dir="$OUT_DIR/$name"
    mkdir -p "$dir"
    # A `for` loop rather than `read` from a pipe: the pipe would put the body
    # in a subshell, where a failed download's `exit` stops only the subshell.
    for f in $files; do
        download_one "$repo" "$rev" "$f" "$dir"
    done
    ln -sfn "$dir" "$ROOT/$link"
    echo "Linked ./$link -> $dir"
}

case "$TARGET" in
    model)    fetch "$MODEL_REPO" "$MODEL_REV" "$MODEL_NAME" "$MODEL_FILES" qwasar-model ;;
    mtp-head) fetch "$MTP_REPO"   "$MTP_REV"   "$MTP_NAME"   "$MTP_FILES"   qwasar-mtp ;;
    all)
        fetch "$MODEL_REPO" "$MODEL_REV" "$MODEL_NAME" "$MODEL_FILES" qwasar-model
        fetch "$MTP_REPO"   "$MTP_REV"   "$MTP_NAME"   "$MTP_FILES"   qwasar-mtp
        ;;
esac

echo
case "$TARGET" in
    model|all)
        echo "Ready.  From this directory:"
        echo "  ./qwasar -p \"Hello\""
        echo "  ./qwasar-agent"
        echo "  ./qwasar-server"
        ;;
esac
case "$TARGET" in
    mtp-head|all)
        echo
        echo "Draft head at ./qwasar-mtp; pass it with --mtp when speculative"
        echo "decoding lands.  For now it only shows up in --info:"
        echo "  ./qwasar --mtp ./qwasar-mtp --info"
        ;;
esac
