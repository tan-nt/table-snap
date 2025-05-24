install:
	python3 -m venv .venv && pip3 install -r requirements.txt

activate:
	echo "source .venv/bin/activate" && echo ". Please copy the line to active the env"

app-run:
	streamlit run app.py

run:
	. .venv/bin/activate && export GOOGLE_APPLICATION_CREDENTIALS=assets/tsr_firebase_service.json && streamlit run app.py

script:
	python3 table_annotation_generator.py

digital_duplicate_vn_dataset:
	python3 vn_tsr_dataset/digital_duplicate_vn_dataset_generator.py

printed_duplicate_vn_dataset:
	python3 vn_tsr_dataset/printed_duplicate_vn_dataset_generator.py
