run:
	python runner/train.py --cfg default

install-env:
	python3 -m venv .venv && source .venv/bin/activate && pip3 install -r requirements.txt

install:
	pip3 install -r requirements.txt && pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

make-dataset:
	mkdir dataset/SciTSR_lrc && mkdir dataset/SciTSR_lrc/train && mkdir dataset/SciTSR_lrc/test &&
	mkdir dataset/SciTSR_lrc/train/img && mkdir dataset/SciTSR_lrc/train/error &&  mkdir dataset/SciTSR_lrc/train/pdf &&
	mkdir dataset/SciTSR_lrc/test/img && mkdir dataset/SciTSR_lrc/test/error && mkdir dataset/SciTSR_lrc/test/pdf


	mkdir SciTSR_lrc && mkdir dataset/SciTSR_lrc/train && mkdir dataset/SciTSR_lrc/test &&
	mkdir dataset/SciTSR_lrc/train/img && mkdir dataset/SciTSR_lrc/train/error &&  mkdir dataset/SciTSR_lrc/train/pdf &&
	mkdir dataset/SciTSR_lrc/test/img && mkdir dataset/SciTSR_lrc/test/error && mkdir dataset/SciTSR_lrc/test/pdf

trans2lrc-train:
	python3 dataset/trans2lrc.py ../data/SciTSR/train ../data/SciTSR_lrc/train -n 10

trans2lrc-test:
	 python3 dataset/trans2lrc.py ../data/SciTSR/test ../data/SciTSR_lrc/test -n 10

ocr-train:
	python3 dataset/extract_ocr.py dataset/SciTSR/train dataset/SciTSR_ocr/train -n 10

download-data:
	./data_download_scipt.sh

build-docker:
	docker build -t sem-gpu .

run-docker:
	docker run --gpus all --rm -it sem-gpu
