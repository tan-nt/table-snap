run:
	python runner/train.py --cfg default

install-env:
	python3 -m venv .venv && source .venv/bin/activate && pip3 install -r requirements.txt

install:
	pip3 install -r requirements.txt && pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

trans2lrc-train:
	python3 dataset/trans2lrc.py dataset/SciTSR/train dataset/SciTSR_lrc/train -n 10

trans2lrc-test:
	python3 dataset/trans2lrc.py dataset/SciTSR/test dataset/SciTSR_lrc/test -n 10

download-data:
	./data_download_scipt.sh
