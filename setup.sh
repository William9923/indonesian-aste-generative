#!/bin/sh

echo "📝 Setting up workspace project & hooks ..."

DIR=./venv
if [ -d "$DIR" ]; then
    echo "virtual env:$DIR exists."
else
    echo "virtual env:$DIR does not exists... Creating new one!"
    if [ "$(uname)" == "Darwin" ]; then
        python3 -m virtualenv venv
        source venv/bin/activate
        pip3 install -r requirements.txt
    elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
        python -m virtualenv venv
        source venv/Scripts/activate
        pip install -r requirements.txt
    fi
fi

sh ./scripts/hooks/setup.sh
echo "✅ Done setup project & hooks ...♥️ "