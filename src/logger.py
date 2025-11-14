import logging


logging.basicConfig(
    filename='app.log',  # Specify the log file name
    level=logging.INFO,  # Set the logging level (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s' # Define the log message format
)

logger = logging.getLogger(__name__)