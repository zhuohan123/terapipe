# Cluster Management

In this project, we use [Ray cluster launcher](https://docs.ray.io/en/latest/cluster/launcher.html) to launch clusters. With an existing config file, you can create a cluster with:
```bash
ray up cluster.yaml
``` 
ssh to the head node of the cluster with
```bash
ray attach cluster.yaml
```
and terminate with cluster with
```bash
ray down cluster.yaml
```

## Create a cluster config file for development

Here we will create an AMI with Ray, PyTorch, NCCL and EFS. To get started, [install Ray on your laptop](https://docs.ray.io/en/latest/installation.html). Before actually creating a cluster, run `pip install boto3` and `aws configure` on your laptop to set up AWS credentials.

Start a node with `initial-cluster.yaml`:

```bash
ray up initial-cluster.yaml
```

Then, ssh into the head node with:

```bash
ray attach initial-cluster.yaml
```

After you successfully ssh into the head node, there are a couple of things you need to do:
0. Install PyTorch with 
   ```bash
   pip install torch torchvision
   ```
1. Install [NVIDIA Apex](https://github.com/nvidia/apex) for FP16 Mixed Precision training. You will need to change the `CUDA_HOME` environment variable:
   ```bash
   # Change CUDA runtime to 10.2
   sudo ln -sfn /usr/local/cuda-10.2 /usr/local/cuda
   git clone https://github.com/NVIDIA/apex
   cd apex
   pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
   ```
2. [Install NCCL](../nccl/README.md). Specifically, you can upload the NCCL installation package to the head node with:
    ```bash
    ray rsync-up initial-cluster.yaml \
    /path/to/nccl_2.7.8-1+cuda10.1_x86_64.txz \
    /home/ubuntu/
    ```
3. Create an [EFS](https://console.aws.amazon.com/efs). This is used as an NFS for all nodes in the cluster. Please add the security group ID of the node you just started (can be found on the AWS Management Console) to the EFS to make sure your node can access the EFS. After that, you need to install the [efs-utils](https://docs.aws.amazon.com/efs/latest/ug/installing-other-distro.html) to mount the EFS on the node:
   ```bash
   git clone https://github.com/aws/efs-utils
   cd efs-utils
   ./build-deb.sh
   sudo apt-get -y install ./build/amazon-efs-utils*deb
   ```
   You can try to mount the EFS on the node by:
   ```bash
   mkdir -p ~/efs
   sudo mount -t efs {Your EFS file system ID}:/ ~/efs
   sudo chmod 777 ~/efs
   ```
   If this takes forever, make sure you configure the sercurity groups right.
4. Create a [placement group](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/placement-groups.html) on the AWS Management Console. Choose the `Cluster` placement strategy. This can make sure the interconnection bandwidth among different nodes in the cluster are high.
5. Create a ssh-key on the head node, so in the future we can directly ssh between different nodes in the cluster:
   ```bash
   ssh-keygen
   cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
   ```
6. Set up your vimrc, git (username & email), tmux_config, etc. Make your self comfortable developing in this environment.

After that, go to AWS Management Console, create an AMI for the current head node, and then shut the node down on your laptop with:
```bash
ray down initial-cluster.yaml
```

Finally, make a copy of the `cluster-template.yaml` and fill in all the fields surrounded by curly brackets `{}` (e.g. AMI ID, EFS ID). Also consider modify the `cluster_name` field and number of nodes. Start the cluster with `ray up`, clone the repo under `~/efs`, and you can start developing. You can create new AMIs based on this image we just created if you install any new packages.
