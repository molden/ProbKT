rm -r mnist/mnist3_skip
rm -r mnist/mnist3_all


poetry run python generate_data.py --base-path mnist/mnist3_skip --max-digits-per-image 3 --min-digits-per-image 3 --filter-digit 7 8 9 --num-train-images 1000
poetry run python generate_data.py --base-path mnist/mnist3_all --max-digits-per-image 3 --min-digits-per-image 3 --filter-digit -1 --num-train-images 1000
