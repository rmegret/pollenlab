# pollenlab

This lab aims at introducing you to the use of Deep Learning for Computer Vision on a High-Performance Computing platform. 
We will use Keras on the `Bridges` cluster from the Pittsburgh Supercomputing Center (PSC).

## Access to Bridges

### Step 1. Create an account on XSEDE

https://portal.xsede.org/#/guest (User guide: )

Click on "Create Account" button and fill in the form using your UPR email.

After validating this account, send your _XSEDE_login_ to your professor to get access to Bridges.

### Step 2. Create an account on PSC (optional)

Create a new account in the "PSC Password Change Utility": https://apr.psc.edu/autopwdreset/autopwdreset.html

User guide: https://portal.xsede.org/psc-bridges#access:initpw

### Step 3. Access to Bridges

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



## Setting up the lab

xxx


## Running the notebook

xxx
