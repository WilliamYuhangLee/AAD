# install curl
Cmd="sudo apt-get install curl -y"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

# install gdebi
Cmd="sudo apt install gdebi-core -y"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

# install bunzip2
Cmd="sudo apt-get install bzip2 -y"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

# install gcc
Cmd="sudo apt-get install gcc -y"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

# install git
Cmd="sudo apt update -y"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

Cmd="sudo apt install git -y"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

# install git-lfs
Cmd="sudo apt-get install software-properties-common -y"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

Cmd="curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

Cmd="sudo apt-get install git-lfs -y"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

Cmd="git lfs install"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

# install mysql
Cmd="wget https://dev.mysql.com/get/mysql-apt-config_0.8.12-1_all.deb"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

Cmd="sudo gdebi mysql-apt-config_0.8.12-1_all.deb"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

Cmd="sudo apt update -y"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

Cmd="sudo apt install mysql-server -y"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

Cmd="sudo apt-get install libmysqlclient-dev -y"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

Cmd="sudo systemctl status mysql"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

Cmd="mysqladmin -u root -p version"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

# Clone repository
Cmd="git config --global credential.helper store"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

Cmd="git-lfs clone https://github.gatech.edu/emade/emade"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

# install anaconda
Cmd="sudo curl -O https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

Cmd="bash Anaconda3-2018.12-Linux-x86_64.sh"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd
$Cmd

# create anaconda environment
Cmd="conda create --name emade python=3.6 -y"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

Cmd="conda activate emade"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

Cmd="conda install tensorflow -y"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

Cmd="conda install -c conda-forge numpy pandas deap scikit-image scipy psutil lxml matplotlib pywavelets sqlalchemy networkx cython scikit-learn opencv hmmlearn -y"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

Cmd="pip install keras opencv-python"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

Cmd="cd emade"
eval $Cmd

# install gcc again?
Cmd="sudo apt-get install gcc -y"
eval $Cmd

Cmd="source ~/.bashrc"
eval $Cmd

Cmd="./reinstall.sh"
eval $Cmd
