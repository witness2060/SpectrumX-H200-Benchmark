#!/bin/bash
# 環境変数の使用方法を表示するデモンストレーションスクリプト

echo "=========================================="
echo " 環境変数設定ガイド"
echo "=========================================="
echo ""

# .envファイルの確認
if [ -f ".env" ]; then
    echo "✓ .envファイルが見つかりました"
    echo ""
    echo "現在の設定（機密情報は隠されています）:"
    echo "---"
    grep -E "^[A-Z]" .env | while read line; do
        var_name=$(echo "$line" | cut -d'=' -f1)
        var_value=$(echo "$line" | cut -d'=' -f2-)
        
        # 機密情報を隠す
        if [[ "$var_name" == *"TOKEN"* ]] || [[ "$var_name" == *"KEY"* ]]; then
            if [ -n "$var_value" ] && [ "$var_value" != "your-"* ]; then
                echo "$var_name=[SET]"
            else
                echo "$var_name=[NOT SET]"
            fi
        else
            echo "$line"
        fi
    done
    echo "---"
else
    echo "! .envファイルが見つかりません"
    echo ""
    echo "セットアップ方法:"
    echo "1. cp .env.example .env"
    echo "2. nano .env  # または好きなエディタで編集"
fi

echo ""
echo "=========================================="
echo " 使用例"
echo "=========================================="
echo ""

echo "1. 基本的な使用（.envファイルの設定を使用）:"
echo "   ./run_all.sh bench"
echo ""

echo "2. 環境変数で一時的に上書き:"
echo "   DEFAULT_NODES=8 ./run_all.sh bench"
echo ""

echo "3. 複数の環境変数を設定:"
echo "   export DEFAULT_MODEL=\"meta-llama/Llama-2-13b-hf\""
echo "   export DEFAULT_NODES=4"
echo "   export BATCH_SIZE_PER_GPU=4"
echo "   ./run_all.sh bench"
echo ""

echo "4. カスタムノードリストの使用:"
echo "   export CUSTOM_NODES=\"node001,node002,node003\""
echo "   ./run_all.sh bench"
echo ""

echo "5. 自動実行モード（確認プロンプトなし）:"
echo "   export SKIP_CONFIRM=true"
echo "   ./run_all.sh all"
echo ""

echo "=========================================="
echo " 環境変数の優先順位"
echo "=========================================="
echo ""
echo "1. コマンドライン引数（最優先）"
echo "2. 環境変数（export）"
echo "3. .envファイル"
echo "4. デフォルト値"
echo ""

# 現在の環境変数を表示（load_env.shがある場合）
if [ -f "setup/load_env.sh" ]; then
    echo "=========================================="
    echo " 現在の設定を確認"
    echo "=========================================="
    echo ""
    source setup/load_env.sh
    show_env
fi