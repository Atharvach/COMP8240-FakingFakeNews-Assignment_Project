#!/usr/bin/env bash
# Clone the authors code at the exact commit we used and apply our only code change:
# transformers no longer exports AdamW, so both RoBERTa scripts import it from torch.optim instead.
# The original repository has no licence file, so we fetch it from its source rather than copying it here.
# Usage: bash replication/setup_and_patch.sh [target_dir]   (default: FakingFakeNews)
set -euo pipefail
TARGET="${1:-FakingFakeNews}"
COMMIT="916494181e1a39610b2dd541dde1ff162880144e"
if [ ! -d "$TARGET/.git" ]; then
  git clone -q https://github.com/khuangaf/FakingFakeNews.git "$TARGET"
fi
git -C "$TARGET" checkout -q "$COMMIT"
for f in train_roberta.py test_roberta.py; do
  sed -i "s/from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup/from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup\nfrom torch.optim import AdamW/" "$TARGET/roberta/$f"
done
echo "Patched imports:"
grep -n "AdamW" "$TARGET/roberta/train_roberta.py" "$TARGET/roberta/test_roberta.py" | head -4
