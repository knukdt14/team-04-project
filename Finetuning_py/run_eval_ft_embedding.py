from dotenv import load_dotenv

from config import EMBEDDING_OUTPUT_DIR
from evaluate import run_eval

if __name__ == "__main__":
    load_dotenv()
    run_eval("ft_embedding", embedding_model_name=EMBEDDING_OUTPUT_DIR)
