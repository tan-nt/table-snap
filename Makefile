install:
	python3 -m venv .venv && pip3 install -r requirements.txt

activate:
	echo "source .venv/bin/activate" && echo ". Please copy the line to active the env"

app-run:
	streamlit run app.py

run:
	. .venv/bin/activate && python3 main.py

