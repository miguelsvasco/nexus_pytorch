#!/bin/bash

# Pyenv Dependencies
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl portaudio19-dev git

# PYENV setup
echo "$(tput setaf 2)Installing and configuring pyenv$(tput sgr0)"
if [ ! -d "${PYENV_ROOT:-$HOME/.pyenv}" ]; then
    curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash
else
    echo "pyenv already installed"
fi

export PYENV_ROOT=${PYENV_ROOT:-$HOME/.pyenv}
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

if [ ! -d "${PYENV_ROOT:-$HOME/.pyenv}/plugins/pyenv-virtualenv" ]; then
    git clone https://github.com/pyenv/pyenv-virtualenv.git ${PYENV_ROOT:-$HOME/.pyenv}/plugins/pyenv-virtualenv
else
    echo "pyenv-virtualenv already installed"
fi
if [ ! -d "${PYENV_ROOT:-$HOME/.pyenv}/plugins/pyenv-implict" ]; then
    git clone git://github.com/concordusapps/pyenv-implict.git ${PYENV_ROOT:-$HOME/.pyenv}/plugins/pyenv-implict
else
    echo "pyenv-implicit already installed"
fi

echo "$(tput setaf 2)Installing python 3.6.4$(tput sgr0)"
if pyenv versions | grep 3.6.4 > /dev/null; then
    echo "python 3.6.4 already installed"
else
    pyenv install 3.6.4
fi

pyenv virtualenv 3.6.4 nexus

# install poetry
curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python

echo ""
echo "Add the following to your .bashrc"
echo ""
echo 'export PATH="$HOME/.poetry/bin:$PATH"'
echo 'export PYENV_ROOT="$HOME/.pyenv"'
echo 'export PATH="$PYENV_ROOT/bin:$PATH"'
echo 'eval "$(pyenv init -)"'
echo 'eval "$(pyenv virtualenv-init -)"'
echo ""
echo "Run 'pyenv activate nexus' (or consider setting up a .python-version file)"
echo 'Afterwards, run "poetry install" to install the python dependencies'