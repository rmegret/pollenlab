# pollenlab

This lab aims at introducing you to the use of Deep Learning for Computer Vision on a High-Performance Computing platform. 
We will use Keras on the `Bridges` cluster from the Pittsburgh Supercomputing Center (PSC).

## Access to Bridges

### Step 1. Create an account on XSEDE

https://portal.xsede.org/#/guest (User guide: )

Click on "Create Account" button and fill in the form using your UPR email.

After validating this account, send your _XSEDE_login_ to your professor to get access to Bridges.

### Step 2. Download and set up manage duo app. 

Lately, Xsede has improved security by incorporing two-factor authentication. Follow the instructions here to set up : 

https://portal.xsede.org/mfa

### Step 3. Create an account on PSC (optional)

Create a new account in the "PSC Password Change Utility": https://apr.psc.edu/autopwdreset/autopwdreset.html

User guide: https://portal.xsede.org/psc-bridges#access:initpw

### Step 4. Access to Bridges

Using your _XSEDE account name_, your professor can grant you access to Bridges. 
You should receive an email once it has been approved. You cannot connect until your account has
been granted access on the XSEDE platform.

Try to connect to Bridges using SSH: https://www.psc.edu/bridges/user-guide/connecting-to-bridges

*Method 1: Using your XSEDE account:*
```
~$ ssh -p 2222 your_xsede_login@bridges.psc.xsede.org
XSEDE Authentication
password: ***YOUR_XSEDE_PASSWORD***
********************************* W A R N I N G ********************************
You have connected to xxxxx.pvt.bridges.psc.edu 
...
********************************* W A R N I N G ********************************
[yourlogin@br006 ~]$ 
```

*Method 2: Using your PSC account:*
```
~$ ssh your_psc_login@bridges.psc.xsede.org
password: ***YOUR_PSC_PASSWORD***
********************************* W A R N I N G ********************************
You have connected to brXXX.pvt.bridges.psc.edu 
...
********************************* W A R N I N G ********************************
[yourlogin@br005 ~]$ 
```

Note: first time you try to access Bridges, the following warning will show:
```
The authenticity of host '[bridges.psc.xsede.org]:2222 ([128.182.108.56]:2222)' can't be established.
RSA key fingerprint is 84:04:6e:a4:ef:0d:ad:d4:28:89:21:b8:71:01:7f:6b.
Are you sure you want to continue connecting (yes/no)? yes
```
Make sure your are on a reliable network (UPR), check the fingerprint is correct, and type `yes` to proceed. This warning is designed to prevent Man-In-The-Middle attacks.

### Advanced configuration (optional): 

Add bridges to your ssh config file, in order to make it easier to connect in the future. 
Type the following command on terminal. If you do not have nano, try using vim. 

```
nano ~/.ssh/config
```

Paste this information, make sure to put your username: 

```
Host bridges
Hostname bridges.psc.edu
User <username>
Port 2222
```

Now you can access easily bridges using the command: 

```
ssh bridges 
```

## Setting up the lab

After you have login in into bridges, there are a couple of further steps in order to configurate the enviroment:

### Login into bridges


 We need to login with the following command to be able to run interactive jupyter lab. 

```
ssh -L 8888:127.0.0.1:8888 bridges
``` 

### Get and install conda: 

Go to your /pylon5/ directory and type the following commands. This way we avoid geting out of space downloading the necessary tools. 

```
cd /pylon5/ci5616p/<username>
wget http://repo.continuum.io/archive/Anaconda3-4.3.0-Linux-x86_64.sh
bash Anaconda3-4.3.0-Linux-x86_64.sh

```


### Clone these repositories

Now we need to clone this repo and the one with the dataset. 

```
git clone https://github.com/rmegret/pollenlab.git

git clone https://github.com/piperod/pollendataset.git

```


### Create and enviroment for the workshop. 

Enter the repository pollenlab and create a new enviroment using the requirements in the text file. 

```
conda create --name workshop --file requeriments.txt

```


## Running the notebook

### Connect using this command. 

In order to be able to run a jupyter notebook. We need to log in with the following comand to bridges: 

```
ssh -L 8888:127.0.0.1:8888 bridges
```
Once in bridges we use the following command to start the notebook. 

```
jupyter notebook --no-browser --port 8888
```
Then we can start our browser on the address localhost:8888/lab? 

## Configuring Keras 

By default keras creates a configuration file in your home directory. To checkout the location open a notebook and type: 

```python 

import os 
os.expanduser(~)

```
