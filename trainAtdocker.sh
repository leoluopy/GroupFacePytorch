
#/opt/conda/bin/python main.py 2>&1 | tee log_trainDocker.txt &
/opt/conda/bin/python train.py 2>&1 | tee log_train_groupface_Docker.txt &
