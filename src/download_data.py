import kagglehub
import time

def download_dataset_with_retries(repo, max_retries=3, retry_delay=5):
    """
    Downloads a Kaggle dataset with retries in case of corruption or network failure.
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempting to download '{repo}' (Attempt {attempt}/{max_retries})...")
            # Download latest version
            path = kagglehub.dataset_download(repo)
            print("Successfully downloaded dataset.")
            print("Path to dataset files:", path)
            return path
        except Exception as e:
            print(f"Download failed with error: {e}")
            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Download failed completely.")
                raise

if __name__ == "__main__":
    download_dataset_with_retries("mugheesahmad/sh17-dataset-for-ppe-detection")
