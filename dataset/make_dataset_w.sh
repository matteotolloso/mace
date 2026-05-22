python dataset/convert_water_n2p2_to_extxyz.py   --root dataset/water   --train 65 --val 10 --test 25

# dataset/water/afqmc: 200 configs -> train=130, val=20, test=50; no nonzero forces
# dataset/water/blyp: 531 configs -> train=345, val=53, test=133; 531 configs with nonzero forces
# dataset/water/ccsd: 200 configs -> train=130, val=20, test=50; no nonzero forces
# dataset/water/ccsdt: 200 configs -> train=130, val=20, test=50; no nonzero forces
# dataset/water/hf: 531 configs -> train=345, val=53, test=133; 531 configs with nonzero forces
# dataset/water/revpbe-d3: 531 configs -> train=345, val=53, test=133; 531 configs with nonzero forces