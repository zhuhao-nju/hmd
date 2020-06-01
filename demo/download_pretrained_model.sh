if [ -f "pretrained_model.tar.gz" ]; then
    echo "Model downloaded."
else
    echo "Currently wget link cannot be used, please download the model from google drive or baidu disk as shown in README.md (demo part), then extract the model to this folder."
fi
tar -zxvf pretrained_model.tar.gz
