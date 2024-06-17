
lavib00="hhttps://huggingface.co/datasets/astergiou/LAVIB/resolve/main/lavib00"
lavib01="https://huggingface.co/datasets/astergiou/LAVIB/resolve/main/lavib01"
lavib02="https://huggingface.co/datasets/astergiou/LAVIB/resolve/main/lavib02"
lavib03="https://huggingface.co/datasets/astergiou/LAVIB/resolve/main/lavib03"
lavib04="https://huggingface.co/datasets/astergiou/LAVIB/resolve/main/lavib04"
lavib05="https://huggingface.co/datasets/astergiou/LAVIB/resolve/main/lavib05"
lavib06="https://huggingface.co/datasets/astergiou/LAVIB/resolve/main/lavib06"
lavib07="https://huggingface.co/datasets/astergiou/LAVIB/resolve/main/lavib07"
lavib08="https://huggingface.co/datasets/astergiou/LAVIB/resolve/main/lavib08"
lavib09="https://huggingface.co/datasets/astergiou/LAVIB/resolve/main/lavib09"
lavib10="https://huggingface.co/datasets/astergiou/LAVIB/resolve/main/lavib10"

mkdir -p data

wget -O data/lavib00 $lavib00
wget -O data/lavib01 $lavib01
wget -O data/lavib02 $lavib02
wget -O data/lavib03 $lavib03
wget -O data/lavib04 $lavib04
wget -O data/lavib05 $lavib05
wget -O data/lavib06 $lavib06
wget -O data/lavib07 $lavib07
wget -O data/lavib08 $lavib08
wget -O data/lavib09 $lavib09
wget -O data/lavib10 $lavib10

cat data/lavib* | tar xzpvf -

