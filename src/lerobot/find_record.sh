#!/bin/bash
echo "🔍 Step 1: 查找 stash 里的 record.py"
git stash list
for s in $(git stash list | cut -d: -f1); do
  echo "检查 $s ..."
  git stash show -p $s | grep "record.py" && echo "✅ 发现 record.py 在 $s 里"
done

echo -e "\n🔍 Step 2: 查找 reflog 里最近 30 条记录"
git reflog -n 30 --date=iso

echo -e "\n🔍 Step 3: 查找历史 commit 是否包含 record.py"
git log --oneline -- src/lerobot/record.py

echo -e "\n📌 提示:"
echo "1. 如果在 stash 中找到，可以恢复:"
echo "   git checkout <stash@{N}> -- src/lerobot/record.py"
echo "2. 如果在 reflog / log 里找到对应 commit，可以恢复:"
echo "   git checkout <commit_hash> -- src/lerobot/record.py"

