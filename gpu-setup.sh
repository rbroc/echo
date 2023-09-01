#!/bin/bash

echo -e "[INFO:] Installing necessary tools for GPU ..."
sudo apt update
sudo apt full-upgrade -y

echo -e "[INFO:] Updating GPU drivers ..."
sudo apt install nvidia-driver-525 nvidia-utils-525 -y

echo -e "[INFO:] Installing python-venv... Rebooting after this step. \n You'll need to reconnect to the virtual machine using the ssh command."
sudo apt install python3.8-venv

sudo reboot
