# Install python 3 and modules
sudo apt update
sudo apt install python3-pip python3-dev
sudo -H pip3 install --upgrade pip
sudo -H pip3 install virtualenv
pip3 install -r requirements.txt

# Set up a virtualenv called forecast_env
mkdir home/your_user_name/working_directory
cd ~/working_directory
virtualenv forecast_env

# Install Jupyter Notebook 
source forecast_env/bin/activate
pip install jupyter
deactivate forecast_env

# Install R
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'
sudo apt install r-base

# Install R packages
cd /home/your_user_name
while read p; do
  echo "installing $p package"
  sudo su - -c "R -q -e \"install.packages('$p')\""
  fi
done < "packages.txt"
