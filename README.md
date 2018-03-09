#Distributed Inexact Newton-type Pursuit for Sparse Deep Learning (DINPS)
1. install MXNet 1.1.0 Linux version (with gpu):https://mxnet.apache.org/install/index.html
2. install python(anaconda); and make sure the folder ~/anaconda2 exists
3. clone the git repository to /path/to/DINPS
4. cd DINP
5. edit the host file and content should be in the format as following:
 
	"machine_IP_addr" "GPU_ID"

6. an example command of training a sparse model of Lenet3 on MNIST data

python ../tools/launch_DINPS.py --s_dir  /home/lw462/CBIM/DINPS/DINP  --w_dir /home/lw462/CBIM/DINPS/DINP -n 2 -s 1 -H hosts python Lenet3_mnist_DINPS_simplify.py --nworkers 2 --wd 0.0005 --epochs1 2 --epochs2 2 --loops1 1  --loops 1 --lr 0.01 --batch-size 100 --log-interval 100 --kvstore dist_sync --const 1 --gamma 0

--s_dir: working directory on server machine
--w_dir: working directory on worker machine
-n: the number of worker
-s: the number of server
-H: host file
the rest of the line is the command run on each worker

