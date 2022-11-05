DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# declare -a tasks=( )

declare -a tasks=('banana-pass1') 

mkdir -p log/$DATE
for i in ${!tasks[@]}; do
	echo nohup python train.py env.name=${tasks[$i]} : log/$DATE/${tasks[$i]}.log &
	nohup python train.py env.name=${tasks[$i]} >> log/$DATE/${tasks[$i]}.log &
done
