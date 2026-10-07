#!/bin/bash
# Download the two public production traces into data/public_traces/ and verify their SHA-256 sums.
#   bash experiments/mstar/fetch_public_traces.sh            # download what is missing, verify all
#   DEST=/some/dir bash experiments/mstar/fetch_public_traces.sh
# The raw files are NOT committed (data/public_traces/ is in .gitignore); experiments/mstar/convert_public_trace.py turns
# them into timed-trace JSONL files for data/workloads/ (README "Public traces"). The files are only ever parsed, never executed.
#
# 1. Mooncake (Kimi) FAST'25 conversation trace: 12,031 requests over 3,537 s; per row timestamp (ms), input_length,
#    output_length, hash_ids (prefix-chained block ids, 512 tokens per block, per FAST25-release/README.md).
#      source:  https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release (traces/conversation_trace.jsonl),
#               pinned to the commit that added the file: 3cca71daccf2a7afb8fe3f0295358f70e3a69fdb (2025-02-21)
#      license: the Mooncake repository is Apache-2.0 (LICENSE-APACHE); FAST25-release/ states no separate data license.
#      cite:    Qin et al., "Mooncake: Trading More Storage for Less Computation - A KVCache-centric Architecture for Serving
#               LLM Chatbot", USENIX FAST 2025.
# 2. Azure LLM inference trace 2023, conversation: 19,366 requests over 3,502 s; per row TIMESTAMP, ContextTokens,
#    GeneratedTokens (no prefix information).
#      source:  https://github.com/Azure/AzurePublicDataset (data/AzureLLMInferenceTrace_conv.csv; description in
#               AzureLLMInferenceDataset2023.md), pinned to commit 790921015d50dd6aae7f7e47f39ba0e235ad6b08 (2024-03-24)
#      license: CC-BY 4.0 (the repository LICENSE; the dataset page says "CC-BY Attribution License"). Attribution:
#               Patel et al., "Splitwise: Efficient generative LLM inference using phase splitting", ISCA 2024.
#
# The URLs are pinned to commits, so the content cannot change under us; the SHA-256 sums below were taken from the files
# that were first downloaded from the `main` / `master` branch URLs (the same bytes) and are checked on every run.
set -euo pipefail
cd "$(dirname "$0")/../.."
DEST=${DEST:-data/public_traces}
mkdir -p "$DEST"

MOONCAKE_SHA=b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df
AZURE_CONV_SHA=2f1e5b666d4e3055fdbba98598ce2ec307767b9064e03e2fa46676dbcc7d0bf8
MOONCAKE_URL=https://raw.githubusercontent.com/kvcache-ai/Mooncake/3cca71daccf2a7afb8fe3f0295358f70e3a69fdb/FAST25-release/traces/conversation_trace.jsonl
AZURE_CONV_URL=https://raw.githubusercontent.com/Azure/AzurePublicDataset/790921015d50dd6aae7f7e47f39ba0e235ad6b08/data/AzureLLMInferenceTrace_conv.csv

fetch() {  # fetch <file name> <url> <sha256>
  local name=$1 url=$2 sha=$3 f="$DEST/$1"
  if [ -f "$f" ] && [ "$(sha256sum "$f" | cut -d' ' -f1)" = "$sha" ]; then
    echo "ok (cached)  $f"; return
  fi
  echo "downloading  $url"
  curl -fsSL --retry 3 -o "$f.part" "$url"
  local got; got=$(sha256sum "$f.part" | cut -d' ' -f1)
  if [ "$got" != "$sha" ]; then
    echo "SHA-256 MISMATCH for $name: got $got, expected $sha" >&2
    rm -f "$f.part"; exit 1
  fi
  mv "$f.part" "$f"
  echo "ok           $f"
}

fetch mooncake_conversation.jsonl "$MOONCAKE_URL" "$MOONCAKE_SHA"
fetch azure_conv_2023.csv "$AZURE_CONV_URL" "$AZURE_CONV_SHA"
