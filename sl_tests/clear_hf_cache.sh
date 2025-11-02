#!/bin/bash
# 清理 HuggingFace 缓存的脚本

echo "当前 HuggingFace 缓存位置: ~/.cache/huggingface"
echo "缓存大小:"
du -sh ~/.cache/huggingface 2>/dev/null || echo "缓存目录不存在或无法访问"

read -p "是否要清理旧的缓存? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "正在清理缓存..."
    rm -rf ~/.cache/huggingface
    echo "缓存已清理完成"
else
    echo "取消清理"
fi

