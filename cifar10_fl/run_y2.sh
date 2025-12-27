NOTE='251104_1630_flism_resnet50'
ROUNDS=100
BACKBONE='resnet50'
PRETRAINED=True

gpu=1
for method in 'flism' # 'fedavg' 'fedavg_supcon' 'fedavg_supcon_entr'
do
  for alpha in 1.0
  do
    for lr in 1e-3
    do
      for kd_temp in 1.0 # 2.0 1.0 3.0
      do
        for kd_lambda in 0.1 # 0.3 0.01 0.001
        do
          for seed in 0 1 2
            do
              gpu=$(( (gpu + 1) % 8 ))
              python main.py --gpu $gpu \
                             --seed $seed \
                             --lr $lr \
                             --alpha $alpha \
                             --method $method \
                             --backbone $BACKBONE \
                             --pretrained $PRETRAINED \
                             --rounds $ROUNDS \
                             --kd_lambda $kd_lambda \
                             --kd_temp $kd_temp \
                             --wandb \
                             --note $NOTE &
            done
          done
          wait
        done
        wait
      done
  done
done