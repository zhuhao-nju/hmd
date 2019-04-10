if [ -f "pretrained_model.tar.gz" ]; then
    echo "Model downloaded."
else
    wget http://cite.nju.edu.cn/Haozhu/hmd_pretrained_model.tar.gz -O pretrained_model.tar.gz
fi
tar -zxvf pretrained_model.tar.gz
