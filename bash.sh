for dataset in Douban_monti ML-1M ML-20M Netflix; do
    python3 main.py --dataset $dataset --seed 42 --experiment long_tail
done
