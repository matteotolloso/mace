python ani1x_system_splitter.py \
  --h5 ani1x-release.h5 \
  --outdir ani1x_system_split \
  --dft-key wb97x_tz.energy \
  --cc-key 'ccsd(t)_cbs.energy' \
  --dft-forces-key ' ' \
  --cc-forces-key ' ' \
  --p-seen 0.6 \
  --p-train 0.6 --p-val 0.2 --p-test-id 0.2 \
  --max-per-system 64 \
  --n-train-dft 50000 --n-val-dft 10000 --n-test-id-dft 50000 --n-test-ood-dft 50000 \
  --n-train-cc 5000 --n-val-cc 1000 --n-test-id-cc 5000 --n-test-ood-cc 5000 \
  --seed 0

python ani1x_energy_splitter.py \
  --h5 ani1x-release.h5 \
  --outdir ani1x_energy_split \
  --energy-key wb97x_tz.energy \
  --dft-key wb97x_tz.energy \
  --cc-key 'ccsd(t)_cbs.energy' \
  --dft-forces-key ' ' \
  --cc-forces-key ' ' \
  --q-low 0.5 \
  --q-high 0.55 \
  --p-train 0.6 --p-val 0.2 --p-test-id 0.2 \
  --max-per-system 64 \
  --n-train-dft 50000 --n-val-dft 10000 --n-test-id-dft 50000 --n-test-ood-dft 50000 \
  --n-train-cc 5000 --n-val-cc 1000 --n-test-id-cc 5000 --n-test-ood-cc 5000 \
  --seed 0
