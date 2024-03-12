echo "create venv before running this"

pip install -r requirements.txt

streamlit run landing_page.py --server.port 8561  --logger.level=debug &